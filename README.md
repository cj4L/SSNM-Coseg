# [AAAI20] Deep Object Co-segmentation via Spatial-Semantic Network Modulation(Oral paper)
## Authors: [Kaihua Zhang](http://kaihuazhang.net/), [Jin Chen](https://github.com/cj4L), [Bo Liu](https://scholar.google.com/citations?user=2Fe93n8AAAAJ&hl=en), [Qingshan Liu](https://scholar.google.com/citations?user=2Pyf20IAAAAJ&hl=zh-CN)
* PDF: [arXiv](https://arxiv.org/abs/1911.12950)

## Abstract  
&emsp;Object co-segmentation is to segment the shared objects in multiple relevant images, which has numerous applications in computer vision. This paper presents a spatial and semantic modulated deep network framework for object co-segmentation. A backbone network is adopted to extract multi-resolution image features. With the multi-resolution features of the relevant images as input, we design a spatial modulator to learn a mask for each image. The spatial modulator captures the correlations of image feature descriptors via unsupervised learning. The learned mask can roughly localize the shared foreground object while suppressing the background. For the semantic modulator, we model it as a supervised image classification task. We propose a hierarchical second-order pooling module to transform the image features for classification use. The outputs of the two modulators manipulate the multi-resolution features by a shift-and-scale operation so that the features focus on segmenting co-object regions. The proposed model is trained end-to-end without any intricate post-processing. Extensive experiments on four image co-segmentation benchmark datasets demonstrate the superior accuracy of the proposed method compared to state-of-the-art methods.

## Overview of our method
![](https://github.com/cj4L/SSNM-Coseg/raw/master/pic/network.png)

&emsp;We propose a spatial-semantic modulated deep network for object co-segmentation. Image features extracted by a backbone network are used to learn a spatial modulator and a semantic modulator. The outputs of the modulators guide the image features up-sampling to generate the co-segmentation results. The network parameter learning is formulated into a multi-task learning task, and the whole network is trained in an end-to-end manner.

&emsp;For the spatial modulation branch, an unsupervised learning method is proposed to learn a mask for each image. With the fused multi-resolution image features as input, we formulate the mask learning as an integer programming problem. Its continuous relaxation has a closed-form solution. The learned parameter indicates whether the corresponding image pixel corresponds to foreground or background.

&emsp;In the semantic modulation branch, we design a hierarchical second-order pooling (HSP) operator to transform the convolutional features for object classification. Spatial pooling (SP) is shown to be able to capture the high-order feature statistical dependency. The proposed HSP module has a stack of two SP layers. They are dedicated to capturing the long-range channel-wise dependency of the holistic feature representation. The output of the HSP layer is fed into a fully-connected layer for object classification and used as the semantic modulator.

## Datasets
&emsp;In order to compare the deep learning methods in recent years fairly, we conduct extensive evaluations on four widely-used benchmark datasets including sub-set of [MSRC](https://www.microsoft.com/en-us/research/project/image-understanding/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fprojects%2Fobjectclassrecognition%2F), [Internet](http://people.csail.mit.edu/mrub/ObjectDiscovery/), sub-set of [iCoseg](http://chenlab.ece.cornell.edu/projects/touch-coseg/), and [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/). Among them:  
* The sub-set of MSRC includes 7 classes: bird, car, cat, cow, dog, plane, sheep, and each class contains 10 images.  
* The Internet has 3 categories of airplane, car and horse. Each class has 100 images including some images with noisy labels.  
* The sub-set of iCoseg contains 8 categories, and each has a different number of images.  
* The PASCAL-VOC is the most challenging dataset with 1037 images of 20 categories selected from the PASCAL-VOC 2010 dataset.  

## Qualitative results
<p align="center">
  <img src="https://github.com/cj4L/SSNM-Coseg/raw/master/pic/unseen.png" width="80%" height="80%">  
</p>
<p align="center">
  <img src="https://github.com/cj4L/SSNM-Coseg/raw/master/pic/seen.png" width="80%" height="80%">  
</p>

## Quantitative results
<p align="center">
  <img src="https://github.com/cj4L/SSNM-Coseg/raw/master/pic/MSRC.png" width="50%" height="50%">  
</p>
<p align="center">
  <img src="https://github.com/cj4L/SSNM-Coseg/raw/master/pic/Internet.png" width="70%" height="70%">  
</p>
<img src="https://github.com/cj4L/SSNM-Coseg/raw/master/pic/iCoseg.png" width="100%" height="100%">  
<img src="https://github.com/cj4L/SSNM-Coseg/raw/master/pic/PASCALVOC.png" width="100%" height="100%">  


#### Schedule
- [x] Create github repo (2019.11.18)
- [x] Release arXiv pdf (2019.12.2)
- [ ] All results (soon)
- [ ] Test and Train code (soon)
