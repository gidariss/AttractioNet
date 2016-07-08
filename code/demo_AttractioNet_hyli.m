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
gpu_id = 1;
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
result_name = 'attentioNet_provided_model_July_7';

root_folder = '/home/hyli/dataset/imagenet16';
gt_path = [root_folder '/ILSVRC2016_DET_bbox_aug'];
im_path = [root_folder '/ILSVRC2016_DET_aug'];
mkdir_if_missing([root_folder '/proposals/' result_name]);

% evaluation code path

%**************************************************************************
%*************************** RUN AttractioNet *****************************
cls_dir = dir(im_path);
cls_dir = cls_dir(3:end);

k = 10;
for i = 14:17%length(cls_dir)
    
    mkdir_if_missing([root_folder '/proposals/' result_name '/' cls_dir(i).name]);
    % don't list the whole images, the list should be based on the
    % annotation file, which is slightly fewer.
    im_dir = dir([gt_path '/' cls_dir(i).name '/*.xml']);
    
    if i == 14
        start = 1395;
    else
        start = 1;
    end
    for j = start:length(im_dir)
        
        % in some case, the original image does not exist! just skip it.
        % 'ski/gettyimg_495794302.jpg'
        try
            image = imread([im_path '/' cls_dir(i).name '/' ...
                im_dir(j).name(1:end-3) 'jpg']);
        catch
            continue;
        end
        bbox_proposals = AttractioNet(model, image, box_prop_conf);
        boxes = bbox_proposals;
        save([root_folder '/proposals/' result_name '/' cls_dir(i).name ...
            '/' im_dir(j).name(1:end-3) 'mat'], 'boxes');
        
        %         bbox_proposals_top_k = bbox_proposals(1:k,1:4);
        %         figure(1); clf(1);
        %         imagesc(image);
        %         drawBBoxes(bbox_proposals_top_k, 'LineWidth', 2);
        %         title(sprintf('top %d box proposal',k));
        %
        %         %rec1 = VOCreadxml('ILSVRC2012_val_00000001.xml');
        %         rec = VOCreadxml([gt_path '/' cls_dir(i).name '/' ...
        %             im_dir(j).name(1:end-3) 'xml']);
        %         try
        %             gt_all_objects = rec.annotation.object;
        %             gt_bbox = extractfield(gt_all_objects, 'bndbox');
        %             gt_bbox = cellfun(@(x) str2double(struct2cell(x))', ...
        %                 gt_bbox, 'UniformOutput', false);
        %             gt_bbox = cell2mat(gt_bbox');
        %
        %             hold on;
        %             drawBBoxes(gt_bbox, 'LineWidth', 3, 'EdgeColor', 'r');
        %             hold off;
        %         catch
        %             % no object in this image, pass it
        %             continue;
        %         end
        tic_toc_print('cls: (%d/%d)\t\timg: (%d/%d)\n', ...
            i, length(cls_dir), j, length(im_dir));
    end
end

caffe.reset_all();