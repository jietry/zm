# BADENet
Code for the paper：Boundary-aware dual edge convolution network for indoor point cloud semantic segmentation

## Introduction
For point cloud semantic segmentation, it makes sense to focus on the boundary information of transition regions. However, most existing excellent point cloud segmentation networks tend to overlook boundary information, resulting in mis-classified transition regions and confusion in feature representation, and thus poor semantic object recognition. For this reason, we propose a network called BADENet. Specifically, the backbone network is first optimized to make the extracted features more robust. Second, a boundary-aware module(BAM) is utilized to focus on the target boundary in the transition region. This module contains a boundary point prediction module(BPPM) and a feature aggregation module(FAM). The boundary point prediction module predicts the points belonging to the target boundary by learning the neighborhood point features. The feature aggregation module performs discriminative aggregation of point cloud features in the local neighborhood under the guidance of the target boundary. In addition, an effective dual edge convolution module(DECM) is designed to model the graph topology, which captures fine-grained geometric information through edge convolution operations, further improves the semantic feature recognition capability, and significantly enriches the contextual information. ## Installation

## Installation
The code is based on [PointNet](https://github.com/charlesq34/pointnet)， [PointNet++](https://github.com/charlesq34/pointnet2) and [BA-GEM](https://github.com/JchenXu/BoundaryAwareGEM).Please install [TensorFlow](https://www.tensorflow.org/install/), and follow the instruction in [PointNet++](https://github.com/charlesq34/pointnet2) to compile the customized TF operators.  
The code has been tested with Python 2.7, TensorFlow 1.9.0, CUDA 9.0 and cuDNN 7.3 on Ubuntu 16.04.

## Usage
### ScanetNet DataSet Segmentation

Download the ScanNetv2 dataset from [here](http://www.scan-net.org/), and see `scannet/README` for details of preprocessing.

To train a model to segment Scannet Scenes:

```
CUDA_VISIBLE_DEVICES=0 python train_ScanNet.py --model BADENet-ScanNet --log_dir pointconv_scannet_ --batch_size 8
```

After training, to evaluate the segmentation IoU accuracies:

```
CUDA_VISIBLE_DEVICES=0 python evaluate_ScanNet.py --model BADENet-ScanNet --batch_size 8 --model_path pointconv_scannet_%s --ply_path DataSet/ScanNetv2/scans
```
Modify the model_path to your .ckpt file path and the ply_path to the ScanNetv2 ply file.
### S3DIS DataSet Segmentation
 Data download and process:
We provide the processed files, you can download S3DIS data <a href="https://1drv.ms/u/s!AjxFyWxg5usOajIvRkNnDLOnT3M?e=mmhCMf">here</a>  . To prepare your own S3DIS Dataset HDF5 files, refer to <a href="https://github.com/charlesq34/pointnet">PointNet</a>, you need to firstly  download <a href="http://buildingparser.stanford.edu/dataset.html">3D indoor parsing dataset version 1.2</a> (S3DIS Dataset) and convert original data to data label files by 

```bash
python collect_indoor3d_data.py
```
Finally run
```bash
python gen_indoor3d_h5.py
```
to downsampling and generate HDF5 files. You can change the number of points in the downsampling by modify this file.

When you have finished download processed data files or have prepared HDF5 files by yourself, to fill in your data path in the `train.py`. Then start training by:

```bash
cd models
python train.py
```
For S3DIS dataset, we tested on the area 5 by default. 

#### Testing

After training, you can test model by:

```bash
python test.py --ckpt  your_ckpt_file  --ckpt_meta your_meta_file
```

Note that the `best_seg_model` chosen by `test.py` is only depend on overall accuracy(OA), maybe mIoU and mAcc value is not the highest. Because   the overall accuracy is not necessarily proportional to the mean IoU. You can test all saved model by:

## License
This repository is released under MIT License (see LICENSE file for details).
