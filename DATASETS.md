## Preparing and using the PASCAL and COCO datasets
In case you need to set up and use the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and/or [COCO](http://mscoco.org/) datasets (e.g. generating or evaluating AttractioNet proposals) then follow the instructions here.

###Preparing COCO dataset
1. Download the images and annotation/info files of the COCO detection task datasets (train 2014, val 2014 and test 2015) from [here](http://mscoco.org/dataset/#download).
2. Place the COCO images and annotation/info files in your local machine with the following structutre:
   ```Shell  
   
   # Images:
   $datasets/MSCOCO/images/train2014/ # train 2014 images directory
   $datasets/MSCOCO/images/val2014/ # val 2014 images directory
   $datasets/MSCOCO/images/test2015/ # test 2015 images directory
   $datasets/MSCOCO/images/test-dev2015/ # test-dev 2015 images directory

   # Annotation/info .json files:
   $datasets/MSCOCO/annotations/instances_train2014.json 
   $datasets/MSCOCO/annotations/instances_val2014.json 
   $datasets/MSCOCO/annotations/image_info_test2015.json
   $datasets/MSCOCO/annotations/image_info_test-dev2015.json
   ```
   where `$datasets` is the directory in your local machine that you usually use for storing all your datasets and `$datasets/MSCOCO` is the parent directory of the COCO dataset. Note that the `$datasets/MSCOCO/images/test-dev2015/` directory could be just a symbolic link to the `$datasets/MSCOCO/images/test2015/` directory:
   ```Shell
   
   ln -sf $datasets/MSCOCO/images/test2015 $datasets/MSCOCO/images/test-dev2015  
   ```
3. Create a symbolic link of the `$datasets` directory at `$AttractioNet/datasets`:
   ```Shell
   
   ln -sf $datasets $AttractioNet/datasets  
   ```
4. Clone the [COCO API](https://github.com/pdollar/coco) in your local machine and then create a symbolic link of its [MatlabAPI](https://github.com/pdollar/coco/tree/master/MatlabAPI) directory at `$AttractioNet/code/MatlabAPI`:  
   ```Shell
   
    # $COCOApi: directory where the COCO API will be cloned   
    git clone https://github.com/pdollar/coco.git $COCOApi   
    ln -sf $COCOApi/MatlabAPI $AttractioNet/code/MatlabAPI    
    ```      
5. Finally, open Matlab from the `$AttractioNet/` directory and run the `script_prepare_COCO_matlab_data_files.m` script:  
   ```Shell
   
    $ cd $AttractioNet  
    $ matlab   
    
    # Matlab command line enviroment
    >> script_prepare_COCO_matlab_data_files   
    ``` 
   Note that the above command will create an extra directory with Matlab files on the location `$datasets/MSCOCO/matlab_files`.

###Preparing PASCAL dataset
1. Download the VOC datasets and VOCdevkit:
   ```Shell
   
   # VOC2007 DATASET
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar # VOC2007 train+val set
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar # VOC2007 test set
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar # VOC2007 devkit
    # VOC2012 DATASET
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # VOC2012 train+val set
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar  # VOC2012 devkit
   ```
2. Untar the VOC2007 tar files in a directory named `$datasets/VOC2007/VOCdevkit` and the VOC2012 tar files in a directory named `$datasets/VOC2012/VOCdevkit`:
   ```Shell
   
   # VOC2007 data:
    mkdir $datasets/VOC2007
    mkdir $datasets/VOC2007/VOCdevkit
    tar xvf VOCtrainval_06-Nov-2007.tar  -C $datasets/VOC2007/VOCdevkit
    tar xvf VOCtest_06-Nov-2007.tar -C $datasets/VOC2007/VOCdevkit
    tar xvf VOCdevkit_08-Jun-2007.tar -C $datasets/VOC2007/VOCdevkit
   # VOC2012 data:
    mkdir $datasets/VOC2012
    mkdir $datasets/VOC2012/VOCdevkit
    tar xvf VOCtrainval_11-May-2012.tar -C $datasets/VOC2012/VOCdevkit
    tar xvf VOCdevkit_18-May-2011.tar -C $datasets/VOC2012/VOCdevkit
   ```
   They should have the following structure:
   ```Shell  
   
   # VOC2007 structure:
   $datasets/VOC2007/VOCdevkit/ # VOC2007 development kit
   $datasets/VOC2007/VOCdevkit/VOCcode/ # VOC2007 development kit code
   $datasets/VOC2007/VOCdevkit/VOC2007/ # VOC2007 images, annotations, etc 
   # VOC2012 structure:
   $datasets/VOC2012/VOCdevkit/ # VOC2012 development kit
   $datasets/VOC2012/VOCdevkit/VOCcode/ # VOC2012 development kit code
   $datasets/VOC2012/VOCdevkit/VOC2012/ # VOC2012 images, annotations, etc 
   ```
   where `$datasets` is the directory in your local machine that you usually use for storing all your datasets.
3. Create a symbolic link of the `$datasets` directory at `$AttractioNet/datasets`:
   ```Shell
   
   ln -sf $datasets $AttractioNet/datasets  
   ```

###Generating and evaluating AttractioNet box proposals on COCO and PASCAL dataset
For a demo on how to generate and evaluate AttractioNet box proposals on the COCO and/or PASCAL datasets see the [demo_extract_AttractioNet_proposals_from_dataset.m](https://github.com/gidariss/AttractioNet/blob/master/code/demo_extract_AttractioNet_proposals_from_dataset.m) script.
