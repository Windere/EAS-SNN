## Introduction
  Event cameras, with their high dynamic range and temporal resolution, are ideally suited for object detection, especially under scenarios with motion blur and challenging lighting conditions. However, while most existing approaches prioritize optimizing spatiotemporal representations with advanced detection backbones and early aggregation functions, the crucial issue of adaptive event sampling remains largely unaddressed. Spiking Neural Networks (SNNs), which operate on an event-driven computing paradigm through sparse spike communication, emerge as a natural fit for addressing this challenge in event-based detection.  In this study, we discover that the neural dynamics of spiking neurons align closely with the behavior of an ideal temporal event sampler. Motivated by this insight, we propose a novel adaptive sampling module that leverages recurrent convolutional SNNs enhanced with temporal memory, facilitating a fully end-to-end learnable framework for event-based detection. Additionally, we introduce Residual Potential Dropout (RPD) and Spike-Aware Training (SAT) to regulate voltage distribution and address performance degradation encountered in spike-based sampling modules. Through rigorous testing on neuromorphic datasets for event-based detection, our approach demonstrably surpasses existing state-of-the-art spike-based methods, achieving superior performance with significantly fewer parameters and time steps. For instance, our method achieves a 4.5\% mAP improvement on the GEN-1 dataset, while requiring 38\% fewer parameters. Moreover, the applicability and effectiveness of our adaptive sampling methodology extend beyond SNNs, as demonstrated through further validation on conventional non-spiking detection models.


## Usage
* First, install all required packages and cd to the tools directory.
1. Commands for training spiking YOLOX-S (non-spiking FPN + non-spiking HEAD) on the GEN-1 dataset
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  python train_event.py -n e-yolox-s -d 4 -b 64  -expn gen1-arsnn-snn-attach-t4k5r0th1-yolox-s-b72-aug.4-sum0-c3-d2-fix-0219-1  max_epoch 30 data_num_workers 4 T 3 eval_interval 10 embedding arsnn basic_lr_per_img 0.000015625 seed 80 data_name gen1 data_dir /data2/wzm/dataset/GEN1/raw/ num_classes 2 scheduler fixed spike_attach True  thresh 1 readout sum embedding_depth 2 embedding_ksize 5 write_zero True  use_spike True spike_fn atan
```

2.  Commands for training spiking YOLOX-S (spiking FPN + spiking HEAD) on the GEN-1 dataset
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  python train_event.py -n e-yolox-s -d 4 -b 54  -expn gen1-arsnn-fsnnv2-attach-t4k5r0th1-yolox-s-b72-aug.4-sum0-c3-d2-fix-0220-1  max_epoch 30 data_num_workers 4 T 3 eval_interval 10 embedding arsnn basic_lr_per_img 0.00001851 seed 80 data_name gen1 data_dir /data2/wzm/dataset/GEN1/raw/ num_classes 2 scheduler fixed spike_attach True  thresh 1 readout sum embedding_depth 2 embedding_ksize 5 write_zero True  use_spike full_spike_v2 spike_fn atan
```

3. Commands for training spiking YOLOX-S (spiking FPN + HEAD) on the GEN-1 dataset
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  python train_event.py -n e-yolox-s -d 4 -b 58  -expn gen1-arsnn-fsnn-attach-t4k5r0th1-yolox-s-b72-aug.4-sum0-c3-d2-fix-0220-1  max_epoch 30 data_num_workers 4 T 3 eval_interval 10 embedding arsnn basic_lr_per_img 0.00001724 seed 80 data_name gen1 data_dir /data2/wzm/dataset/GEN1/raw/ num_classes 2 scheduler fixed spike_attach True  thresh 1 readout sum embedding_depth 2 embedding_ksize 5 write_zero True  use_spike full_spike spike_fn atan