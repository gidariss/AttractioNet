## *Attend Refine Repeat: Active Box Proposal Generation via In-Out Localization*

### Introduction

The *AttractioNet* code implements the following arxiv paper:    
**Title:**      "Attend Refine Repeat: Active Box Proposal Generation via In-Out Localization"    
**Authors:**     Spyros Gidaris, Nikos Komodakis    
**Institution:** Universite Paris Est, Ecole des Ponts ParisTech    
**ArXiv Link:**  http://arxiv.org/abs/1511.07763   
**Code:**        https://github.com/gidariss/AttractioNet 

**Abstract:**  
The problem of computing category agnostic bounding box proposals is utilized as a core component in many computer vision tasks and thus has lately attracted a lot of attention. In this work we propose a new approach to tackle this problem that is based on an active strategy for generating box proposals that starts from a set of seed boxes, which are uniformly distributed on the image, and then progressively moves its attention on the promising image areas where it is more likely to discover well localized bounding box proposals. We call our approach *AttractioNet* and a core component of it is a CNN-based category agnostic object location refinement module that is capable of yielding accurate and robust bounding box predictions regardless of the object category. We extensively evaluate our *AttractioNet* approach on several image datasets (i.e. COCO, PASCAL, ImageNet detection and NYU-Depth V2 datasets) reporting on all of them state-of-the-art results that surpass the previous work in the field by a significant margin and also providing strong empirical evidence that our approach is capable to generalize to unseen categories. Furthermore, we evaluate our *AttractioNet* proposals in the context of the object detection task using a VGG16-Net based detector and the achieved detection performance on COCO manages to significantly surpass all other VGG16-Net based detectors while even being competitive with a heavily tuned ResNet-101 based detector.

### Main Results
 Test set                 | AR@10 | AR@100 | AR@1000  | AR@Small |  AR@Medium | AR@Large |time/img
-------------------------:|:-----:|:------:|:--------:|:--------:|:----------:|:--------:|:------:
 COCO 2014 val.           | 0.326 | 0.532  | 0.660    | 0.317    | 0.621      | 0.771    |1.63secs
 VOC 2007 test            | 0.547 | 0.740  | 0.848    | 0.575    | 0.666      | 0.788    |1.63secs
 ImageNet detection task val.| 0.412 | 0.618  | 0.748    |  -       | -          | -        |1.63secs
 NYU-Depth V2             | 0.159 | 0.389  | 0.579    | 0.205    | 0.419      | 0.498    |1.63secs

### Citing AttractioNet

If you find AttractioNet useful in your research, please consider citing:   
> @article{gidaris2016attend,  
  title={Attend Refine Repeat: Active Box Proposal Generation via In-Out Localization},  
  author={Gidaris, Spyros and Komodakis, Nikos},   
  journal={arXiv preprint arXiv:1511.07763},   
  year={2015}
}  

and 

> @inproceedings{gidaris2016locnet,  
  title={LocNet: Improving Localization Accuracy for Object Detection},  
  author={Gidaris, Spyros and Komodakis, Nikos},   
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on},  
  year={2016}  
}  

### License
This code is released under the MIT License (refer to the LICENSE file for details).  

### Contents
1. [Requirements](#requirements)   
2. [Installation](#installation)   
3. [Downloading pre-computed bounding box proposals](#downloading-pre-computed-bounding-box-proposals)   
4. [Preparing and using the COCO and PASCAL datasets](#preparing-and-using-the-coco-and-pascal-datasets)  

### Requirements

**Hardware.**  In order to use AttractioNet for extracting bounding box proposals from an image you will require a GPU with at least 4 Gbytes of memory

**Software.**       
1. A modified, in order to support AttractioNet, version of Caffe [[link]](https://github.com/gidariss/caffe_LocNet/tree/AttractioNet) (you must use the AttractioNet branch).  
2. The cuDNN(-v5) library during Caffe installation.    
3. Matlab (tested with R2014b).
  
### Installation 

1. Download and install this modified version of [Caffe](https://github.com/gidariss/caffe_LocNet/tree/AttractioNet),   initially developed to supprot [LocNet](https://github.com/gidariss/caffe_LocNet) and now extended to support *AttractioNet* (use the AttractioNet branch for the later). Clone Caffe in your local machine:         
   ```Shell
    
    # $caffe_AttractioNet: directory where Caffe will be cloned 
    git clone https://github.com/gidariss/caffe_LocNet.git $caffe_AttractioNet  
    # switch on the AttractioNet branch  
    git checkout AttractioNet                              
    ```         
  Then follow the Caffe and Matcaffe installation instructions [here](http://caffe.berkeleyvision.org/installation.html). **Note** that you have to install Caffe with the cuDNN(-v5) library.   
2. Clone the *AttractioNet* code in your local machine:  
   ```Shell
   
    # $AttractioNet: directory where AttractioNet will be cloned    
    git clone https://github.com/gidariss/AttractioNet $AttractioNet  
    ```   
  From now on, the directory where *AttractioNet* is cloned will be called `$AttractioNet`.  
3. Create a symbolic link of [Caffe](https://github.com/gidariss/caffe_LocNet/tree/AttractioNet) installatation directory at `$AttractioNet/external/caffe_AttractioNet`:  
   ```Shell
   
    # $AttractioNet: directory where AttractioNet is cloned   
    # $caffe_AttractioNet: directory where caffe is cloned and installed    
    ln -sf $caffe_AttractioNet $AttractioNet/external/caffe_AttractioNet   
    ```      
4. Download the [***AttractioNet pre-trained model***](https://drive.google.com/file/d/0BwxkAdGoNzNTV2N3RjN5dXNpWVE/view?usp=sharing). Note that the provided model is actually the fast version of AttractioNet model that is described on section 3.1.3 of technical report. After downloading, gunzip and untar the .tar.gz archive file with the AttractioNet model files on the directory `$AttractioNet/models-exps/AttractioNet_Model` by running:   
   ```Shell
   
   tar xvfz AttractioNet_Model.tar.gz -C $AttractioNet/models-exps/    
   ```   
5.  open matlab from the `$AttractioNet/` directory and run the `AttractioNet_build.m` script:  
   ```Shell
   
    $ cd $AttractioNet  
    $ matlab   
    
    # matlab command line enviroment
    >> AttractioNet_build   
    ``` 
    Do not worry about the warning messages. They also appear on my machine.  

### Demo
After having complete the installation, you will be able to use *AttractioNet* for extracting bounding box proposals from any image. For a demo see the [demo_AttractioNet.m](https://github.com/gidariss/AttractioNet/blob/master/code/demo_AttractioNet.m) script.  Note that you will require a GPU with at least 4 Gbytes of memory in order to run the demo. 

### Downloading pre-computed bounding box proposals
We provide pre-computed bounding box proposals --- using the same AttractioNet model that we provide here --- for the following datasets:

**PASCAL VOC:**    
- [**Trainval 2007 (135.8 MB)**](https://mega.nz/#!6ksHGTiI!R2h-j-tQNh9FSGP_kji02zdDbPK2lhEyWcMAKkH_ej4) 
- [**Test 2007 (134.2 MB)**](https://mega.nz/#!ag9UUCZJ!Fw9i9ZBuFjP_olj7wjL3tvZrcvkXdvQmvvTfwza1Iro)
- [**Trainval 2012 (312.6 MB)**](https://mega.nz/#!Ll1EWCxS!MbfmjMalOn6k2f0jF26ioJ7x91vEfuQu0ud5-rQFmDk) 
- [**Test 2012 (297.8 Mb)**](https://mega.nz/#!K1NzTQxD!-s38tOeu6C7hO4wEyMB_8CqmNZMM8mj5hiQcXA7ZlX8)   

**MS-COCO:**  
- [**Train 2014 (2.21 GB)**](https://mega.nz/#!b08mCYLR!8njSxoq946-SZSYTHkgMtwsTsH6FYBwPJAGBQsLE3eQ)  
- [**Val 2014 (1.08 GB)**](https://mega.nz/#!ypsDCZiC!MwlQ-pLV9Y_VYO9w469uylqiAfrr8UgsTwrbBZf1YA4)  
- [**Test 2015 (2.18 GB)**](https://mega.nz/#!6oNDAJhK!E-1mO7Md8Ln5Bnm4OgLg28ZgSpOhvOKINQ42U2Ydktg)  

Each package contains the AttractioNet box proposals of the corresponding data set stored using a separate box proposal file per image. Specifically, the box proposals of each image are stored in Matlab files (.mat files) using the same filenames as those that the images have. Each Matlab file contains a single data field, called *boxes*, that is a `K x 5` single precision array with the box proposals of the corresponding image (where `K` is the number of box proposals). Each row of the *boxes* array contains a single bounding box proposal represented by the 5 values `[x0,y0,x1,y1,obj]`, where `(x0,y0)` and `(x1,y1)` are its top-left and bottom-right coordinates correspondingly (specifically, 1-based pixel coordinates) and `obj` is its objectness score. Note that the box proposals are already sorted w.r.t. their objectness score in decreasing order.

### Preparing and using the COCO and PASCAL datasets
In case you need to set up and use the COCO and/or PASCAL datasets (e.g. generating or evaluating AttractioNet proposals) then follow the instructions on the [DATASET.md](https://github.com/gidariss/AttractioNet/blob/master/DATASETS.md) file.
