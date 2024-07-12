#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch

# from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets import NCALTECH_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
    get_rank,
    RecordHook
)


def per_class_AR_table(coco_eval, class_names=NCALTECH_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=NCALTECH_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


# todo: split the inference and evaluation
class EventEvaluator:
    """
    AP Evaluation class for Customized Event Dataset.  Refer to the NCaltech101 dataset as an example.
    """

    def __init__(
            self,
            dataloader,
            img_size: int,
            confthre: float,
            nmsthre: float,
            num_classes: int,
            testdev: bool = False,
            per_class_AP: bool = True,
            per_class_AR: bool = True,
            snn_reset=False
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.snn_reset = snn_reset

    def evaluate(
            self, model, distributed=False, half=False, trt_file=None,
            decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        output_data, gt_dict = defaultdict(), defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            logger.log("Using TensorRT engine for event-based inference")
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
        assert self.dataloader.dataset.map_val and not self.dataloader.dataset.random_aug, "the dataset must be " \
                                                                                           "set as the mode of " \
                                                                                           "map val. and not " \
                                                                                           "random_aug "
        # bboxes = xyxy2xywh(bboxes)
        #
        # for ind in range(bboxes.shape[0]):
        #     # label = self.dataloader.dataset.class_ids[int(cls[ind])]
        #     label = int(cls[ind])
        #     pred_data = {
        #         "image_id": int(img_id),
        #         "category_id": label,
        #         "bbox": bboxes[ind].numpy().tolist(),
        #         "score": scores[ind].numpy().item(),
        #         "segmentation": [],
        #     }  # COCO json format
        #     data_list.append(pred_data)
        data_btime = time.time()
        for cur_iter, (imgs, labels, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
        ):
            # processing predictions
            with torch.inference_mode(mode=True):
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)

                if self.snn_reset:
                    from spikingjelly.activation_based.functional import reset_net
                    reset_net(model)

                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
                # print("data time {} {} {}, infer time:  {}, nms time {}".format(
                #     {k: v / self.dataloader.dataset.profile['count'] for k, v in
                #      self.dataloader.dataset.profile.items()},
                #     (sum([v / self.dataloader.dataset.profile['count'] for k, v in
                #           self.dataloader.dataset.profile.items()]) - 1) * self.dataloader.batch_size,
                #     start - data_btime, infer_end - start,
                #     nms_end - infer_end))
            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

            # processing ground truth
            for i, (label, height, width, img_id) in enumerate(zip(labels, info_imgs[0], info_imgs[1], ids)):
                bboxes = label[:, :4]
                cls = label[:, 4]
                gt_dict.update({
                    int(img_id): {
                        "bboxes": [box.numpy().tolist() for box in bboxes],
                        "width": width,
                        "height": height,
                        "category_ids": [
                            # self.dataloader.dataset.class_ids[int(cls[ind])]
                            int(cls[ind])  # do not need for event-based data
                            for ind in range(bboxes.shape[0])
                        ],
                    }
                })
            data_btime = time.time()
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            data_list = gather(data_list, dst=0)
            gt_dict = gather(gt_dict, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            gt_dict = dict(ChainMap(*gt_dict))
            torch.distributed.reduce(statistics, dst=0)
            print("rank {} collect done".format(get_rank()))

        eval_results = self.evaluate_prediction(data_list, gt_dict, statistics)
        print("rank {} eval done".format(get_rank()))
        synchronize()
        print("rank {} sync done".format(get_rank()))

        if return_outputs:
            return eval_results, output_data
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        """
            # return_outputs: if return the dict from image id to bbox info.
            # outputs: [box_1, box_2, ..., box_bz] xyxy
            # make sure the dataloader return the correct size of raw image
            # make sure the mapping from cls id to cls name is stored in the variable self.dataloader.dataset.class_name
        """
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
                outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize back into the original image size
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        # self.dataloader.dataset.class_ids[int(cls[ind])]
                        int(cls[ind])  # todo: fix it as name
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                # label = self.dataloader.dataset.class_ids[int(cls[ind])]
                label = int(cls[ind])  # do not need for event-based data,
                pred_data = {
                    # "image_id": str(int(img_id)),
                    # "category_id": label + 1,
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def getcocoGT2(self, gt_dict):
        # verfiy the function of getcocoGT through the existing coco formatter
        from yolox.utils.util import preprocess_gt
        from pycocotools.coco import COCO
        GT_PATH = os.path.join("./map/", "ground-truth")
        for image_id in gt_dict:
            with open(os.path.join(GT_PATH, str(image_id) + ".txt"), "w") as new_f:
                for box, cat in zip(gt_dict[image_id]["bboxes"], gt_dict[image_id]["category_ids"]):
                    box = np.array(box).astype(np.int64).tolist()
                    new_f.write(
                        "%s %s %s %s %s\n" % (NCALTECH_CLASSES[cat], box[0], box[1], box[0] + box[2], box[1] + box[3]))
        print("Get ground truth result done.")
        GT_JSON_PATH = os.path.join("./map/", 'instances_gt.json')
        with open(GT_JSON_PATH, "w") as f:
            results_gt = preprocess_gt(GT_PATH, NCALTECH_CLASSES)
            json.dump(results_gt, f, indent=4)
        cocoGt = COCO(GT_JSON_PATH)
        return cocoGt

    def getcocoGT(self, gt_dict):
        from pycocotools.coco import COCO
        # 构造 COCO 格式的注释数据
        coco_annotations = {
            "images": [],
            "annotations": [],
            "categories": [{
                "id": id,
                "name": name,
                "supercategory": name
            } for id, name in enumerate(self.dataloader.dataset.class_names)]  # 如果你有类别信息的话，也需要包含在内
        }

        # 添加图像和注释信息
        for image_id in gt_dict:
            image_info = gt_dict[image_id]
            coco_annotations["images"].append({
                "id": image_id,
                "file_name": self.dataloader.dataset.sample_names[image_id],
                "width": image_info["width"],
                "height": image_info["height"]
            })

            for box, cat_id in zip(image_info["bboxes"], image_info["category_ids"]):
                coco_annotations["annotations"].append({
                    "id": len(coco_annotations["annotations"]),
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0
                    # 添加其他注释信息
                })

        # 创建 COCO 对象
        cocoGt = COCO()
        cocoGt.dataset = coco_annotations
        cocoGt.createIndex()
        return cocoGt
        # from pycocotools.coco import COCO
        # if self.testdev:
        #     json.dump(gt_dict, open("./yolox_testdev_gt_2017.json", "w"))
        #     cocoGt = COCO("./yolox_testdev_gt_2017.json")
        # else:
        #     _, tmp = tempfile.mkstemp()
        #     json.dump(gt_dict, open(tmp, "w"))
        #     cocoGt = COCO(tmp)
        # return cocoGt

    # todo: calculate the MAP based on prophese evaluator
    # BBOX_DTYPE = np.dtype({'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
    #                        'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
    #                        'offsets': [0, 8, 12, 16, 20, 24, 28, 32], 'itemsize': 40})
    def evaluate_prediction(self, data_dict, gt_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                ["forward", "NMS", "inference"],
                [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
            )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            # cocoGt = self.dataloader.dataset.coco
            from yolox.utils import wait_for_the_master
            # with wait_for_the_master():
            cocoGt = self.getcocoGT(gt_dict)
            print("Get ground truth result done with cocoGT1.")
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            # todo: try to calculate the mAP without writing to json file. by wzm
            # todo: handle the coco label list in the dataset. by wzm
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
                # try:
                #     from yolox.layers import COCOeval_opt as COCOeval
                # except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info


    def energy_estimation(self,model,exp,half=False, trt_file=None,test_size=None):
        logger.info("\n start energy estimation" )
        import torch.nn as nn
        import copy
        hook_cls = (nn.Conv2d, nn.Linear)
        progress_bar = tqdm if is_main_process() else iter
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor

        def calc_layer_sop(layer,inputs,ac=True):
            sop_ac = 0
            sop_mac = 0
            spike_counts = inputs.sum(0).cuda()
            analog_counts = torch.ones_like(spike_counts)
            aux_layer = copy.deepcopy(layer)
            aux_layer.weight = torch.nn.Parameter(torch.ones_like(aux_layer.weight))
            if aux_layer.bias is not None:
                aux_layer.bias = torch.nn.Parameter(torch.zeros_like(aux_layer.bias))
            if ac:
                sop_ac = aux_layer(spike_counts).sum()
            sop_mac = aux_layer(analog_counts).sum()
            return sop_ac,sop_mac
        def calcSOP():
            num_samples = 0
            tot_som_ac = 0
            tot_som_mac = 0
            module_ac = {'embedding':0,'backbone':0,'fpn':0,'head':0}
            module_mac = {'embedding':0,'backbone':0,'fpn':0,'head':0}
            with torch.no_grad():
                cali_layers = {'embedding': {},'backbone':{},'fpn':{},'head':{}}
                for m in model.named_children():
                    module_name = m[0]
                    if m[0] == 'backbone':
                        for m_bb in m[1].named_children():
                            if m_bb[0] == 'backbone':
                                module_name = 'backbone'
                            else:
                                module_name = 'fpn'
                            for l in m_bb[1].named_modules():
                                if isinstance(l[1], hook_cls):
                                    cali_layers[module_name][l[0]] = l[1]
                    else:
                        for l in m[1].named_modules():
                            if isinstance(l[1], hook_cls):
                                cali_layers[module_name][l[0]] = l[1]

                # data_btime = time.time()
                for cur_iter, (imgs, labels, info_imgs, ids) in enumerate(
                        progress_bar(self.dataloader)
                ):
                    num_samples += len(imgs)
                    imgs = imgs.type(tensor_type)
                    hooks = {'embedding':{},'backbone':{},'fpn':{},'head':{}}
                    for key, module_layers in cali_layers.items():
                        for name,layer in module_layers.items():
                            hooker = RecordHook(to_cpu=True)
                            handler = layer.register_forward_hook(hooker)
                            hooks[key][name] = (handler,hooker)
                    model(imgs)

                    for key,module_layers in cali_layers.items():
                        for name,layer in module_layers.items():
                            handler,hooker = hooks[key][name]
                            if_ac = True
                            if key != 'embedding':
                                assert len(hooker.inputs) == 1
                                hooker.inputs = hooker.inputs[0].reshape([exp.T,-1]+list(hooker.inputs[0].shape[1:]))
                            else :
                                hooker.inputs = torch.stack(hooker.inputs)
                            if name in ('input_conv.0','input_conv.2','gate_conv.2','stem.0.conv.conv','dark2.0.conv.0'):
                                if_ac = False

                            handler.remove()
                            torch.cuda.empty_cache()
                            sop_ac,sop_mac = calc_layer_sop(layer, hooker.inputs, ac=if_ac)
                            module_ac[key]+=sop_ac
                            tot_som_ac += sop_ac
                            module_mac[key] += sop_mac
                            tot_som_mac += sop_mac
                            del hooker
            return module_ac, module_mac, tot_som_ac, tot_som_mac, num_samples

        model = model.eval()
        if half:
            model = model.half()
        if trt_file is not None:
            logger.log("Using TensorRT engine for event-based inference")
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        sop_ac, sop_mac,tot_ac,tot_mac,num_samples = calcSOP()
        print('\nSOP in SNN: {}, SOP in ANN {}'.format(tot_ac /num_samples / 1e9, tot_mac / num_samples / 1e9))
        print('SNN Energy:{}'.format(0.9 * tot_ac /num_samples / 1e9))
        print('ANN Energy:{}'.format(4.6 * tot_mac /num_samples / 1e9))
        for key, value in sop_ac.items():
            print('{}: SOP in SNN: {}, SOP in ANN {}'.format(key,sop_ac[key] / num_samples / 1e9, sop_mac[key] / num_samples / 1e9))
