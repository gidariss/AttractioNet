% demo_AttractioNet demonstrates how to use AttractioNet for extracting the
% bounding box proposals from a single image.


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
box_prop_conf = AttractioNet_get_defaul_conf();
fprintf('AttractioNet configuration params:\n');
disp(box_prop_conf);
model.scales   = box_prop_conf.scales;   
model.max_size = box_prop_conf.max_size; 
fprintf('AttractioNet Model:\n');
disp(model);
%**************************************************************************
%****************************** READ IMAGE ********************************
image_path = fullfile(pwd,'examples','COCO_val2014_000000109798.jpg');
image = imread(image_path);
%**************************************************************************
%*************************** RUN AttractioNet *****************************
bbox_proposals = AttractioNet(model, image, box_prop_conf);
% bbox_proposals is a K x 5 single precision array where K is the
% number of output box proposals (by default K=2000). Each row of the 
% bbox_proposals array is a single bounding box proposal represented by the
% 5 values [x0,y0,x1,y1,obj] where (x0,y0) and (x1,y1) are the top-left 
% and bottom-right coordinates of the bounding box in pixels (one-based 
% coordinates) and obj is the objectness score of the bounding box proposal 
% after the multi-threshold NMS re-ordering. Note that the box proposals
% are already sorted w.r.t. to their objectness score obj in decreasing 
% order. Hence, if for example you want to use the top 100 box proposals 
% then all you have to do is:
% bbox_proposals_top100 = bbox_proposals(1:100,:);

%**************************************************************************
%************************ VISUALIZE BOX PROPOSALS *************************
% visualize the top 10 box proposals 
k = 10;
bbox_proposals_top_k = bbox_proposals(1:k,1:4);
figure(1); clf(1);
imagesc(image);
drawBBoxes(bbox_proposals_top_k,'LineWidth',3);
title(sprintf('top %d box proposal',k));

caffe.reset_all();