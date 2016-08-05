% demo_AttractioNet demonstrates how to use AttractioNet for extracting the
% bounding box proposals from ilsvrc 2015/2014, the standard officical version.

%clc;
clear;
close all;
run('startup');
caffe.reset_all();
% %************************* SET GPU/CPU DEVICE *****************************
% % By setting gpu_id = 1, the first GPU (one-based counting) is used for
% % running the AttractioNet model. By setting gpu_id = 0, then the CPU will
% % be used for running the AttractioNet model. Note that I never tested it
% % my self the CPU.
% gpu_id = 0;
% caffe_set_device( gpu_id );
% caffe.set_mode_gpu();
% %**************************************************************************
% %***************************** LOAD MODEL *********************************
% model_dir_name = 'AttractioNet_Model';
% full_model_dir = fullfile(pwd, 'models-exps', model_dir_name);
% assert(exist(full_model_dir,'dir')>0,sprintf('The %s model directory does not exist',full_model_dir));
% mat_file_name  = 'box_proposal_model.mat';
% model = AttractioNet_load_model(full_model_dir, mat_file_name);
% %**************************************************************************
% %********************** CONFIGURATION PARAMETERS **************************
% disp(' ');
% box_prop_conf = AttractioNet_get_defaul_conf();
%
% box_prop_conf.multiple_nms_test = false;
%
% %box_prop_conf.nms_range = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5];
% box_prop_conf.nms_range = [0.5 0.48 0.45, 0.43 0.4 0.38 0.35 0.33 0.3];
%
% fprintf('AttractioNet configuration params:\n');
% disp(box_prop_conf);
% model.scales   = box_prop_conf.scales;
% model.max_size = box_prop_conf.max_size;
% fprintf('AttractioNet Model:\n');
% disp(model);
%
% %**************************************************************************
% %****************************** READ IMAGE ********************************
% %image_path = fullfile(pwd,'examples','COCO_val2014_000000109798.jpg');
% %image_path = fullfile(pwd,'examples','000029.jpg');
% %image = imread(image_path);
%
% % dataset
% result_name = 'attentioNet_provided_model_July_31_merge';
result_path = './box_proposals/author_provide/val2';
%
% root_folder = '/home/hongyang/dataset/imagenet_det/ILSVRC2014_devkit';
% fid = fopen([root_folder '/data/det_lists/val2.txt'], 'r');
% temp = textscan(fid, '%s%s');
% im_list = temp{1}; clear temp;
% gt_path = [root_folder '/../ILSVRC2013_DET_bbox_val/'];
% im_path = [root_folder '/../ILSVRC2013_DET_val'];
%
% mkdir_if_missing([result_path '/' result_name]);

%**************************************************************************
%*************************** RUN AttractioNet *****************************
proposal_name = 'bbox_props_cands_Aug_1_default.mat';
if ~exist(proposal_name, 'file')
    boxes_all = cell(length(im_list), 1);
    boxes_uncut = cell(length(im_list), 1);
    for i = 1:length(im_list)
        
        image = imread([im_path '/' im_list{i} '.JPEG']);
        [boxes_all{i}, boxes_uncut{i}] = AttractioNet(model, image, box_prop_conf);
        tic_toc_print('img: (%d/%d)\n', i, length(im_list));
    end
    caffe.reset_all();
    save(proposal_name, 'boxes_uncut');
end

ld = load(proposal_name);
raw_aboxes = ld.boxes_uncut;

addpath('/home/hongyang/project/faster_rcnn/functions/rpn');
addpath(genpath('/home/hongyang/project/faster_rcnn/utlis'));
%%
multi_nms_name = 'multi_nms_minus_case';
mkdir([result_path '/' multi_nms_name]);
factor_vec = [1 : -0.1 :0 ];
%nms.scheme = 'no_minus';
nms.scheme = 'minus';
multi_NMS_setting = generate_multi_nms_setting_attend();
imdb.name = 'ilsvrc14_val2';
imdb.flip = false;

for i = 1:length(factor_vec)
    for j = 1:length(multi_NMS_setting)
        
        nms.note = sprintf('multiNMS_fac_%.1f_set_%d_%s', ...
            factor_vec(i), j, nms.scheme);
        cprintf('blue', 'do attentioNet multi-thres NMS, taking a while ...\n');
        
        parfor kk = 1:length(raw_aboxes)
            aboxes{kk} = AttractioNet_postprocess(...
                raw_aboxes{kk}, 'thresholds', -inf, 'use_gpu', true, ...
                'mult_thr_nms',     true, ...
                'nms_iou_thrs',     multi_NMS_setting(j).nms_iou_thrs, ...
                'factor',           factor_vec(i), ...
                'scheme',           nms.scheme, ...
                'max_per_image',    multi_NMS_setting(j).max_per_image);
        end
        save([result_path '/' multi_nms_name '/' nms.note '.mat'], 'aboxes', '-v7.3');
        % compute recall
        recall_per_cls = compute_recall_ilsvrc(...
            [result_path '/' multi_nms_name '/' nms.note '.mat'], 300, imdb);
        mean_recall = 100*mean(extractfield(recall_per_cls, 'recall'));
        
        cprintf('blue', 'multi-thres nms note (%s), mean rec:: %.2f\n\n', nms.note, mean_recall);
        save([result_path '/' multi_nms_name '/' nms.note ...
            sprintf('_recall_%.2f.mat', mean_recall)], 'recall_per_cls');       
    end
end


%% normal nms below
% proposal_path_jot = cell(length(boxes_all{1}), 1);
% imdb.name = 'ilsvrc14_val2';
% imdb.flip = false;
%
% for i = 1:length(boxes_all{1})
%     aboxes = cell(length(im_list), 1);
%     parfor j = 1:length(im_list)
%         aboxes{j} = boxes_all{j}{i};
%     end
%     if i == 1
%         proposal_path_jot{i} = [result_path '/' ...
%             result_name '/boxes_multi_thres_nms.mat'];
%     else
%         proposal_path_jot{i} = [result_path '/' result_name '/' ...
%             sprintf('boxes_nms_%.2f.mat', box_prop_conf.nms_range(i-1))];
%     end
%     save(proposal_path_jot{i}, 'aboxes');
% end
%
% for i = 1:length(boxes_all{1})
%     % compute recall
%     recall_per_cls = compute_recall_ilsvrc(proposal_path_jot{i}, 300, imdb);
%     mean_recall = 100*mean(extractfield(recall_per_cls, 'recall'));
%
%     cprintf('blue', 'i = %d, mean rec:: %.2f\n\n', i, mean_recall);
%     save([proposal_path_jot{i}(1:end-4) sprintf('_recall_%.2f.mat', mean_recall)], 'recall_per_cls');
% end
