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
from yolox.data.datasets import GEN1_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
    get_rank,

)

from yolox.utils.psee_loader.evaluator import PropheseeEvaluator
from yolox.utils import wait_for_the_master


def per_class_AR_table(coco_eval, class_names=GEN1_CLASSES, headers=["class", "AR"], colums=6):
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


def per_class_AP_table(coco_eval, class_names=GEN1_CLASSES, headers=["class", "AP"], colums=6):
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


class PSEEEvaluator:
    """
    AP Evaluation class for Event Dataset with Prophesee protocol.
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
            dataset="gen1",
            downsample_by_2=False,
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
        self.evaluator = PropheseeEvaluator(dataset.lower(), downsample_by_2)

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

            batch_preds = self.convert_to_gt_format(outputs, info_imgs)
            batch_labels = [torch.cat([label, torch.ones_like(label[:, 0:1])], dim=1) for label in labels]
            sample_names = [self.dataloader.dataset.sample_names[id] for id in ids]
            batch_preds = self.convert_to_prophesee_format(batch_preds, sample_names)
            batch_labels = self.convert_to_prophesee_format(batch_labels, sample_names)
            if distributed:
                batch_preds = gather(batch_preds, dst=0)
                batch_preds = list(itertools.chain(*batch_preds))
                batch_labels = gather(batch_labels, dst=0)
                batch_labels = list(itertools.chain(*batch_labels))
            self.evaluator.add_labels(batch_labels)
            self.evaluator.add_predictions(batch_preds)
        print("Having Data?", self.evaluator.has_data())
        if self.evaluator.has_data():
            with wait_for_the_master():
                eval_results = self.evaluator.evaluate_buffer(self.dataloader.dataset.img_size[0],
                                                              self.dataloader.dataset.img_size[1])
                prefix = f'PROHESEE Evaluation/'
                info = ''
                for k, v in eval_results.items():
                    if isinstance(v, (int, float)):
                        value = torch.tensor(v)
                    elif isinstance(v, np.ndarray):
                        value = torch.from_numpy(v)
                    elif isinstance(v, torch.Tensor):
                        value = v
                    else:
                        raise NotImplementedError
                    assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
                    # put them on the current device to avoid this error: https://github.com/Lightning-AI/lightning/discussions/2529
                    info += f'{prefix}{k}  ' + str(value) + " \n"
        predictions = self.evaluator._buffer[self.evaluator.PREDICTIONS]
        self.evaluator.reset_buffer()
        if return_outputs:
            return (eval_results['AP'], eval_results['AP_50'], info), predictions
        return '', info

    def convert_to_gt_format(self, outputs, info_imgs):
        data_list = []
        for (output, img_h, img_w) in zip(
                outputs, info_imgs[0], info_imgs[1]
        ):
            if output is None:
                data_list.append(torch.tensor([[0, 0, 0, 0, 0, 0]]))
                continue
            output = output.cpu()
            bboxes = output[:, 0:4]
            # preprocessing: resize back into the original image size
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            dt_item = torch.cat([bboxes, cls[:, None], scores[:, None]], dim=1)
            data_list.append(dt_item)
        return data_list

    def get_time_from_name(self, name):
        return int(name.split('a')[-1])

    def convert_to_prophesee_format(self, bboxes, sample_names):
        BBOX_DTYPE = np.dtype({'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
                               'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
                               'offsets': [0, 8, 12, 16, 20, 24, 28, 32], 'itemsize': 40})
        box_proph_list = []
        for i, (box, name) in enumerate(zip(bboxes, sample_names)):
            num_box = len(box)
            time = self.get_time_from_name(name)
            box_proph = np.zeros((num_box,), dtype=BBOX_DTYPE)
            if num_box > 0:
                assert box.shape == (num_box, 6)
                box_proph['t'] = np.ones((num_box,), dtype=BBOX_DTYPE['t']) * time
                box_proph['x'] = np.asarray(box[:, 0], dtype=BBOX_DTYPE['x'])
                box_proph['y'] = np.asarray(box[:, 1], dtype=BBOX_DTYPE['y'])
                box_proph['w'] = np.asarray(box[:, 2], dtype=BBOX_DTYPE['w'])
                box_proph['h'] = np.asarray(box[:, 3], dtype=BBOX_DTYPE['h'])
                box_proph['class_id'] = np.asarray(box[:, 4], dtype=BBOX_DTYPE['class_id'])
                box_proph['class_confidence'] = np.asarray(box[:, 5], dtype=BBOX_DTYPE['class_confidence'])
            box_proph_list.append(box_proph)
        return box_proph_list

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
