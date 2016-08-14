% demo_AttractioNet demonstrates how to use AttractioNet for extracting the
% bounding box proposals from ilsvrc 2015/2014, the standard officical version.

%clc;
clear;
close all;
run('startup');
caffe.reset_all();
addpath('../faster_rcnn/functions/rpn');
addpath(genpath('../faster_rcnn/utlis'));

gpu_id = 0;
caffe_set_device( gpu_id );
caffe.set_mode_gpu();
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
box_prop_conf.multiple_nms_test = true;
% probably optimal setting
box_prop_conf.nms_iou_thrs = [.9 : -0.05 : .5];
box_prop_conf.max_per_image = [2000;1800;1620;1458;1312;1180;1062;955;859];
box_prop_conf.nms_range = [0.8 : -0.05 : 0.4];
fprintf('AttractioNet configuration params:\n');
disp(box_prop_conf);
model.scales   = box_prop_conf.scales;
model.max_size = box_prop_conf.max_size;
fprintf('AttractioNet Model:\n');
disp(model);

%**************************************************************************
%****************************** READ IMAGE ********************************
% dataset
result_name = 'provided_model_Aug_14th';


% sub_dataset = 'val1';
imdb.name = 'ilsvrc14_val1_14';
imdb.name = 'ilsvrc14_val1_13';
imdb.name = 'ilsvrc14_pos1k_13';
%imdb.name = 'ilsvrc14_real_test';
% sub_dataset = 'train14';
% imdb.name = 'ilsvrc14_train14';
sub_dataset = strrep(imdb.name, 'ilsvrc14_', '');
% ------------------------------------------
result_path = sprintf('./box_proposals/author_provide/%s', sub_dataset);
mkdir_if_missing([result_path '/' result_name]);
switch imdb.name
    case 'ilsvrc14_train14'
        root_folder = '/home/hongyang/dataset/imagenet_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/train14.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2014_DET_train'];
        extension = '.JPEG';
        imdb.flip = true;
        
    case 'ilsvrc14_val1'
        root_folder = '/home/hongyang/dataset/imagenet_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/val1.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2013_DET_val'];
        extension = '.JPEG';
        imdb.flip = true;
        
    case 'ilsvrc14_val2'
        root_folder = '/home/hongyang/dataset/imagenet_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/val1.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2013_DET_val'];
        extension = '.JPEG';
        imdb.flip = false;
        
        % the following datasets wont compute recall since I am just too lazy
        % to collect their GT info.
    case 'ilsvrc14_val1_14'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/val1_14.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2014_DET_train'];
        extension = '';
        imdb.flip = false;
        
    case 'ilsvrc14_val1_13'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/val1_13.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2013_DET_val'];
        extension = '.JPEG';
        imdb.flip = false;
        
    case 'ilsvrc14_real_test'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/real_test.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2015_DET_test'];
        extension = '.JPEG';
        imdb.flip = false;
        
    case 'ilsvrc14_pos1k_13'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/pos1k_13.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2014_DET_train'];
        extension = '.JPEG';
        imdb.flip = false;
end
if imdb.flip
    test_im_list_flip = cellfun(@(x) [x '_flip'], test_im_list, 'uniformoutput', false);
    test_im_list_new = cell(2*length(test_im_list_flip), 1);
    test_im_list_new(1:2:end) = test_im_list;
    test_im_list_new(2:2:end) = test_im_list_flip;
    test_im_list = test_im_list_new;
end

%**************************************************************************
%*************************** RUN AttractioNet *****************************
whole_proposal_file = fullfile(result_path, result_name, 'boxes_uncut.mat');

if ~exist(whole_proposal_file, 'file')
    
    boxes_all = cell(length(test_im_list), 1);
    boxes_uncut = cell(length(test_im_list), 1);
    
    for i = 1:length(test_im_list)
        
        if i == 1 || i == length(test_im_list) || mod(i, 1000) == 0
            fprintf('extract box, method: %s, dataset: %s, (%d/%d)...\n', ...
                'attractioNet', sub_dataset, i, length(test_im_list));
        end
        image = imread([im_path '/' test_im_list{i} '.JPEG']);
        [boxes_all{i}, boxes_uncut{i}] = AttractioNet(model, image, box_prop_conf);
        
    end
    caffe.reset_all();
    save(whole_proposal_file, 'boxes_uncut');
end

% %% ========= temporal =========
% load(whole_proposal_file);
% box_prop_conf.threshold = -Inf;
% boxes_all = cell(length(test_im_list), 1);
%
% for kk = 1:length(boxes_uncut)
%     if kk == 1 || kk == length(boxes_uncut) || mod(kk, 1000) == 0
%         fprintf('temp, progress: (%d/%d)', kk, length(test_im_list));
%     end
%     bbox_props_out = AttractioNet_postprocess(boxes_uncut{kk}, ...
%         'thresholds',       box_prop_conf.threshold, ...
%         'use_gpu',          true, ...
%         'mult_thr_nms',     length(box_prop_conf.nms_iou_thrs)>1, ...
%         'nms_iou_thrs',     box_prop_conf.nms_iou_thrs, ...
%         'max_per_image',    box_prop_conf.max_per_image);
%
%     if box_prop_conf.multiple_nms_test
%         proposals_per_im = cell(1+length(box_prop_conf.nms_range), 1);
%         proposals_per_im{1} = bbox_props_out;
%
%         for i = 1:length(box_prop_conf.nms_range)
%             proposals_per_im{1+i} = AttractioNet_postprocess(boxes_uncut{kk}, ...
%                 'nms_iou_thrs',     box_prop_conf.nms_range(i), ...
%                 'max_per_image',    2000);
%         end
%     end
%     boxes_all{kk} = proposals_per_im;
% end
%=============================

%% normal nms below
proposal_path_jot = cell(length(boxes_all{1}), 1);
for i = 1:length(boxes_all{1})
    aboxes = cell(length(test_im_list), 1);
    for j = 1:length(test_im_list)
        aboxes{j} = boxes_all{j}{i};
    end
    if i == 1
        proposal_path_jot{i} = [result_path '/' ...
            result_name '/boxes_multi_thres_nms_optimal.mat'];
    else
        proposal_path_jot{i} = [result_path '/' result_name '/' ...
            sprintf('boxes_nms_%.2f.mat', box_prop_conf.nms_range(i-1))];
    end
    save(proposal_path_jot{i}, 'aboxes');
end

% compute recall
for i = 1:length(boxes_all{1})
    recall_per_cls = compute_recall_ilsvrc(proposal_path_jot{i}, 300, imdb);
    mean_recall = 100*mean(extractfield(recall_per_cls, 'recall'));
    
    cprintf('blue', 'i = %d, mean rec:: %.2f\n\n', i, mean_recall);
    save([proposal_path_jot{i}(1:end-4) sprintf('_recall_%.2f.mat', mean_recall)], 'recall_per_cls');
end

%% multi-thres nms
% ld = load(proposal_name);
% raw_aboxes = ld.boxes_uncut;
% multi_nms_name = 'multi_nms_minus_case';
% mkdir([result_path '/' multi_nms_name]);
% factor_vec = [1 : -0.1 :0 ];
% %nms.scheme = 'no_minus';
% nms.scheme = 'minus';
% multi_NMS_setting = generate_multi_nms_setting_attend();
% imdb.name = 'ilsvrc14_val2';
% imdb.flip = false;
% for i = 1:length(factor_vec)
%     for j = 1:length(multi_NMS_setting)
%
%         nms.note = sprintf('multiNMS_fac_%.1f_set_%d_%s', ...
%             factor_vec(i), j, nms.scheme);
%         cprintf('blue', 'do attentioNet multi-thres NMS, taking a while ...\n');
%
%         parfor kk = 1:length(raw_aboxes)
%             aboxes{kk} = AttractioNet_postprocess(...
%                 raw_aboxes{kk}, 'thresholds', -inf, 'use_gpu', true, ...
%                 'mult_thr_nms',     true, ...
%                 'nms_iou_thrs',     multi_NMS_setting(j).nms_iou_thrs, ...
%                 'factor',           factor_vec(i), ...
%                 'scheme',           nms.scheme, ...
%                 'max_per_image',    multi_NMS_setting(j).max_per_image);
%         end
%         save([result_path '/' multi_nms_name '/' nms.note '.mat'], 'aboxes', '-v7.3');
%         % compute recall
%         recall_per_cls = compute_recall_ilsvrc(...
%             [result_path '/' multi_nms_name '/' nms.note '.mat'], 300, imdb);
%         mean_recall = 100*mean(extractfield(recall_per_cls, 'recall'));
%
%         cprintf('blue', 'multi-thres nms note (%s), mean rec:: %.2f\n\n', nms.note, mean_recall);
%         save([result_path '/' multi_nms_name '/' nms.note ...
%             sprintf('_recall_%.2f.mat', mean_recall)], 'recall_per_cls');
%     end
% end
