# EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks
<p align="center">
  <img src="./docs/view.png" width="750">
</p>
This is the official Pytorch implementation of the ECCV 2024 paper [End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks](https://arxiv.org/abs/2403.12574)

 **Summary:** In this study, we discover that the neural dynamics of spiking neurons align closely with the behavior of an ideal temporal event sampler. Motivated by this, we propose a novel adaptive sampling module that leverages recurrent convolutional SNNs enhanced with temporal memory, facilitating a fully end-to-end learnable framework for event-based detection. Additionally, we introduce Residual Potential Dropout (RPD) and Spike-Aware Training (SAT) to regulate potential distribution and address performance degradation encountered in spike-based sampling modules.  


## Installation
The main dependencies are listed below:

| Dependency     | Version |
|----------------|---------|
| spikingjelly   | 0.0.0.0.14 |
| h5py           | 3.8.0   |
| torchvision    | 0.16.1  |
| thop           | 0.1.1   |
| pytorch        | 2.1.1   |
| pycocotools    | 2.0.6   |
| opencv         | 4.7.0   |
| numpy          | 1.26.0  |
| einops         | 0.8.0   |
|python          |          3.10.9|
You can try to install the required packages by running:
```bash 
conda env create -f environment.yml
```
or
```bash
pip install -r pip-requirements.txt
```

## Required Data
1. The raw GEN-1 dataset can be downloaded from [here](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)

2. The raw 1Mpx dataset can be downloaded from [here](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/)

3. The preprocessed 1Mpx dataset by RVT can be downloaded from [here](https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen4.tar)

4. The raw N-Caltech 101 dataset by RVT can be downloaded from [here](https://www.garrickorchard.com/datasets/n-caltech101)

After unzipping the dataset, you should have the following directory structure:
```shell
    # The Splited N-Caltech101 Dataset
    ├── N-Caltech
    │   ├── Caltech101
    │   ├── Caltech101_annotations
    │   ├── test.txt
    │   ├── train.txt
    │   ├── val.txt
    
    #  The raw 1Mpx/Gen1 dataset
    ├── Root Directory
    │   ├── Raw Splitted Dataset
    │   │   ├── train
    │   │   │   ├── EVENT_STREAM_NAME_td.dat
    │   │   │   ├── EVENT_STREAM_NAME_bbox.npy
    │   │   │   └── ...
    │   │   ├── val
    │   │   │   ├── EVENT_STREAM_NAME_td.dat
    │   │   │   ├── EVENT_STREAM_NAME_bbox.npy
    │   │   │   └── ...
    │   │   ├── test
    │   │   │   ├── EVENT_STREAM_NAME_td.dat
    │   │   │   ├── EVENT_STREAM_NAME_bbox.npy
    │   │   │   └── ...
    
    # The processed 1Mpx Dataset
    ├── Root Directory
    │   ├── train
    │   │   ├── EVENT_STREAM_NAME
    │   │   │   ├── event_representations_v2
    │   │   │   │   ├── stacked_histogram_dt=50_nbins=10   
    │   │   │   │   │   ├── event_representations_ds2_nearest.h5  
    │   │   │   │   │   ├── objframe_idx_2_repr_idx.npy 
    │   │   │   │   │   ├── timestamps_us.npy 
    │   │   │   ├── labels_v2
    │   │   │   │   ├── labels.npz   
    │   │   │   │   ├── timestamps_us.npy 
    │   │   │   │   ├──  ...

 ```

## Usage
1. First, install all required packages and cd to the 'tools' directory.
2. Run the following command to train an EAS-SNN model on the GEN-1 dataset:
   * Commands for training spiking YOLOX-S (non-spiking FPN + non-spiking HEAD) on the GEN-1 dataset
       ```bash
       CUDA_VISIBLE_DEVICES=0,1,2,3  python train_event.py -n e-yolox-s -d 4 -b 64 \
       -expn exp_name  max_epoch 30 data_num_workers 4 T 3 eval_interval 10 embedding arsnn \ 
      basic_lr_per_img 0.000015625 seed 80 data_name gen1 data_dir /data2/wzm/dataset/GEN1/raw/ \
     num_classes 2 scheduler fixed spike_attach True  thresh 1 readout sum embedding_depth 2 \
      embedding_ksize 5 write_zero True  use_spike True spike_fn atan
       ```

   * Commands for training spiking YOLOX-S (spiking FPN + HEAD) on the GEN-1 dataset
       ```bash
       CUDA_VISIBLE_DEVICES=0,1,2,3  python train_event.py -n e-yolox-s -d 4 -b 58 \
      -expn exp_name  max_epoch 30 data_num_workers 4 T 3 eval_interval 10 embedding arsnn \
      basic_lr_per_img 0.00001724 seed 80 data_name gen1 data_dir /data2/wzm/dataset/GEN1/raw/ \
      num_classes 2 scheduler fixed spike_attach True  thresh 1 readout sum embedding_depth 2 \
      embedding_ksize 5 write_zero True  use_spike full_spike spike_fn atan
       ```

   * Commands for training spiking YOLOX-S (spiking FPN + spiking HEAD) on the GEN-1 dataset
       ```bash
       CUDA_VISIBLE_DEVICES=0,1,2,3  python train_event.py -n e-yolox-s -d 4 -b 54  -expn exp_name \
      max_epoch 30 data_num_workers 4 T 3 eval_interval 10 embedding arsnn basic_lr_per_img 0.00001851 \ 
     seed 80 data_name gen1 data_dir /data2/wzm/dataset/GEN1/raw/ num_classes 2 scheduler fixed spike_attach True \
      thresh 1 readout sum embedding_depth 2 embedding_ksize 5 write_zero True  use_spike full_spike_v2 spike_fn atan
       ```
3. Run the following command to evaluate an EAS-SNN model on the GEN-1 dataset:
    ```bash
    python eval_event.py -n e-yolox-m -d 4 -b 36 -c ./YOLOX_outputs/$exp_name/best_ckpt.pth --conf 0.001 \
     --eval_proh  data_num_workers 4 embedding arsnn seed 80 data_name gen1 data_dir /data2/wzm/dataset/GEN1/raw/ \
     num_classes 2 Tm 4 T 3 spike_attach True thresh 1 readout sum embedding_depth 2 embedding_ksize 5 \
    write_zero True use_spike full_spike spike_fn atan
     ```
4. Consider to modify the hyperparameter Ts to explore the ability for temporal modelling capacity of SNNs as shown in Fig.4 in the paper.


## Citation Info
If you find this work helpful, please consider citing our paper:
```bibtex
@article{wang2024eas,
  title={EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks},
  author={Wang, Ziming and Wang, Ziling and Li, Huaning and Qin, Lang and Jiang, Runhao and Ma, De and Tang, Huajin},
  journal={arXiv preprint arXiv:2403.12574},
  year={2024}
}
```

## Acknowledgement
This project has adpated code from the following libraries:
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for the detection PAFPN/head
* [Tonic](https://github.com/neuromorphs/tonic) for the event-based representation, like voxel grid
* [RVT](https://github.com/uzh-rpg/RVT/tree/master) for the 1Mpx preprocessed dataset and the Prophesee evaluation tool
* [ASGL](https://github.com/Windere/ASGL-SNN) and [SpikingJelly](https://github.com/fangwei123456/spikingjelly) for the SNN implementation
* [Spikingjelly](https://github.com/fangwei123456/spikingjelly) for event visualization


