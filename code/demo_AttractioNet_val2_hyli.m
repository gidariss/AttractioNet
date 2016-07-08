% demo_AttractioNet demonstrates how to use AttractioNet for extracting the
% bounding box proposals from ilsvrc 2016 and beyond.

%clc;
clear;
close all;
%************************* SET GPU/CPU DEVICE *****************************
% By setting gpu_id = 1, the first GPU (one-based counting) is used for
% running the AttractioNet model. By setting gpu_id = 0, then the CPU will
% be used for running the AttractioNet model. Note that I never tested it
% my self the CPU.
gpu_id = 2;
caffe_set_device( gpu_id );
caffe.reset_all();
%**************************************************************************
%***************************** LOAD MODEL *********************************
model_dir_name = 'AttractioNet_Model';
full_model_dir = fullfile(pwd, 'models-exps', model_dir_name);
assert(exist(full_model_dir,'dir')>0,sprintf('The %s model directory does not exist',full_model_dir));
mat_file_name  = 'box_proposal_model.mat';
model = AttractioNet_load_model(full_model_dir, mat_file_name);
%**************************************************************************
%********************** CONFIGURATION PARAMETERS **************************
disp(' ');
box_prop_conf = AttractioNet_get_defaul_conf();
fprintf('AttractioNet configuration params:\n');
disp(box_prop_conf);
model.scales   = box_prop_conf.scales;
model.max_size = box_prop_conf.max_size;
fprintf('AttractioNet Model:\n');
disp(model);

%**************************************************************************
%****************************** READ IMAGE ********************************
%image_path = fullfile(pwd,'examples','COCO_val2014_000000109798.jpg');
%image_path = fullfile(pwd,'examples','000029.jpg');
%image = imread(image_path);

% dataset
result_name = 'attentioNet_provided_model_July_7_val2';
result_path = './box_proposals/author_provide/val2';

root_folder = '/home/hongyang/dataset/imagenet_det/ILSVRC2014_devkit';
fid = fopen([root_folder '/data/det_lists/val2.txt'], 'r');
temp = textscan(fid, '%s%s');
im_list = temp{1}; clear temp;
gt_path = [root_folder '/../ILSVRC2013_DET_bbox_val/'];
im_path = [root_folder '/../ILSVRC2013_DET_val'];

mkdir_if_missing([result_path '/' result_name]);

%**************************************************************************
%*************************** RUN AttractioNet *****************************
for i = 5001:length(im_list)
    
    image = imread([im_path '/' im_list{i} '.JPEG']);
    bbox_proposals = AttractioNet(model, image, box_prop_conf);
    boxes = bbox_proposals;
    save([result_path '/' result_name '/' im_list{i} '.mat'], 'boxes');
    
    tic_toc_print('img: (%d/%d)\n', i, length(im_list));
end

caffe.reset_all();