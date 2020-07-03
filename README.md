# [AAAI20] Deep Object Co-segmentation via Spatial-Semantic Network Modulation(Oral paper)
## Authors: [Kaihua Zhang](http://kaihuazhang.net/), [Jin Chen](https://github.com/cj4L), [Bo Liu](https://scholar.google.com/citations?user=2Fe93n8AAAAJ&hl=en), [Qingshan Liu](https://scholar.google.com/citations?user=2Pyf20IAAAAJ&hl=zh-CN)
* PDF: [arXiv](https://arxiv.org/abs/1911.12950) or [AAAI20](https://aaai.org/ojs/index.php/AAAI/article/view/6977)

## Abstract  
&emsp;Object co-segmentation is to segment the shared objects in multiple relevant images, which has numerous applications in computer vision. This paper presents a spatial and semantic modulated deep network framework for object co-segmentation. A backbone network is adopted to extract multi-resolution image features. With the multi-resolution features of the relevant images as input, we design a spatial modulator to learn a mask for each image. The spatial modulator captures the correlations of image feature descriptors via unsupervised learning. The learned mask can roughly localize the shared foreground object while suppressing the background. For the semantic modulator, we model it as a supervised image classification task. We propose a hierarchical second-order pooling module to transform the image features for classification use. The outputs of the two modulators manipulate the multi-resolution features by a shift-and-scale operation so that the features focus on segmenting co-object regions. The proposed model is trained end-to-end without any intricate post-processing. Extensive experiments on four image co-segmentation benchmark datasets demonstrate the superior accuracy of the proposed method compared to state-of-the-art methods.

## Examples
<p align="center">
  <img src="https://github.com/cj4L/SSNM-Coseg/raw/master/pic/unseen.png" width="80%" height="80%">  
</p>

## Overview of our method
![](https://github.com/cj4L/SSNM-Coseg/raw/master/pic/network.png)

## Datasets
&emsp;In order to compare the deep learning methods in recent years fairly, we conduct extensive evaluations on four widely-used benchmark datasets including sub-set of [MSRC](https://www.microsoft.com/en-us/research/project/image-understanding/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fprojects%2Fobjectclassrecognition%2F), [Internet](http://people.csail.mit.edu/mrub/ObjectDiscovery/), sub-set of [iCoseg](http://chenlab.ece.cornell.edu/projects/touch-coseg/), and [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/). Among them:  
* The sub-set of MSRC includes 7 classes: bird, car, cat, cow, dog, plane, sheep, and each class contains 10 images.  
* The Internet has 3 categories of airplane, car and horse. Each class has 100 images including some images with noisy labels.  
* The sub-set of iCoseg contains 8 categories, and each has a different number of images.  
* The PASCAL-VOC is the most challenging dataset with 1037 images of 20 categories selected from the PASCAL-VOC 2010 dataset.  

## Results download
* VGG16-backbone: [Google Drive](https://drive.google.com/file/d/14h2XdIB0GR1Zb_0X59T0URgbuogJpR5Z/view?usp=sharing) or [Baidupan code:fxat](https://pan.baidu.com/s/1bG5Biq3omAkBaY9mHRzrZw).
* HRNet-backbone: [Google Drive](https://drive.google.com/file/d/1r8piQHHVosecDJD6DmDZVriUzEfQxCeB/view?usp=sharing) or [Baidupan code:mn2k](https://pan.baidu.com/s/1bht2GhxCBM4XPk9GMahpwg).

## Environment
* Ubuntu 16.04, Nvidia RTX 2080Ti
* Python 3
* PyTorch>=1.0, TorchVision>=0.2.2
* Numpy==1.16.2, Pillow, pycocotools

## Test
* Get or download the dataset we have processed in [Google Drive](https://drive.google.com/file/d/1bo5zE64bQwLUbCUGKDLcRjHei9FBmhfi/view?usp=sharing) or [Baidupan code:ap2u](https://pan.baidu.com/s/1PeIj3YLIde-0BB8raC3KBQ).
* Download VGG16-backbone pretrained model in [Google Drive](https://drive.google.com/file/d/1Vvir1CeuCNQY7GU_Ygh593U5I-KXZWff/view?usp=sharing) or [Baidupan code:eoav](https://pan.baidu.com/s/1VMJoxXPm1n3xupVZb3EiBg).
* Modify the path config in coseg_test.py and run it.

## Train
* Get the COCO2017 Dataset for training the whole network.
* Get the test dataset for val and test phase.
* Download VGG16 pretrained weights in [Google Drive](https://drive.google.com/file/d/1KIWIspVxLRwv8bzOuMn6lY8kStoedToV/view?usp=sharing) or [Baidupan code:3aga](https://pan.baidu.com/s/1iI7Umk7TNiOiI_wL43ipAw). Actually is from PyTorch offical model weights, expect for deleting the last serveral layers.
* Download dict.npy in [Google Drive](https://drive.google.com/file/d/1p15hGN3YwqWMRN4xx5mDIK04OhimpY2z/view?usp=sharing) or [Baidupan code:9ccf](https://pan.baidu.com/s/1n8CKyZWoP1tUr1FtyZouig).
* Modify the path config in main.py and run it.

### Notes
* Following the suggestion of reviewers in AAAI20, we would not release the HRNet-backbone trained model for fairly comparing with others methods. 
* There are some slight differences in the 'Fusion' part of the model but little impact.
* There is a mistake value in Table 2, our HRNet J-index(82.5) in 'Car' in Internet Dataset should be modified with (73.9).

#### Schedule
- [x] Create github repo (2019.11.18)
- [x] Release arXiv pdf (2019.12.2)
- [x] Release AAAI20 pdf (2020.7.3)
- [x] All results (2020.7.3)
- [x] Test and Train code (2020.7.3)
