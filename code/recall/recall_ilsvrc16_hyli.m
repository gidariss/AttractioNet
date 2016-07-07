%clc;
clear; close all;

% revised by Hongyang
% date:         July 7, 2016

% if top_k is not indicated, it will evaluate all proposals
top_k = 300; %2000;
% overlap ratio
ov = 0.5;

%% the provided object proposals
% 'proposals' is a cell-type variable
root_folder = '/media/hongyang/Hongyang_MobileDrive/dataset_in_mobile_drive/ILSVRC2016';
gt_path = [root_folder '/ILSVRC2016_DET_bbox_aug'];

result_name = 'attentioNet_provided_model_July_7';
proposal_path = [root_folder '/proposals/' result_name];
% current available results
cls_dir = dir(proposal_path);
cls_dir = cls_dir(3:end);

%% compute recall
total_cls_num = 200;
ld = load('sup_data/ilsvrc_meta_det.mat');
synsets = ld.synsets_det;
recall_per_cls = [];
recall_per_cls(total_cls_num).name = 'to_be_decide';
for i = 1:total_cls_num
    recall_per_cls(i).wnid = synsets(i).WNID;
    recall_per_cls(i).name = synsets(i).name;
    recall_per_cls(i).total_inst = 0;
    recall_per_cls(i).correct_inst = 0;
    recall_per_cls(i).recall = 0;
end
name_list = extractfield(recall_per_cls, 'name')';

for i = 1:length(cls_dir)
    
    prop_dir = dir([proposal_path '/' cls_dir(i).name '/*.mat']);
    
    for j = 1:length(prop_dir)
        % per image
        
        % load 'boxes'
        load([proposal_path '/' cls_dir(i).name '/' prop_dir(j).name]);
        
        % load GT
        rec = VOCreadxml([gt_path '/' cls_dir(i).name '/' ...
            prop_dir(j).name(1:end-3) 'xml']);
        try
            gt_all_objects = rec.annotation.object;
            gt_bbox = extractfield(gt_all_objects, 'bndbox');
            gt_bbox = cellfun(@(x) str2double(struct2cell(x))', ...
                gt_bbox, 'UniformOutput', false);
            gt_bbox = cell2mat(gt_bbox');
        catch
            % no object in this image, pass it
            continue;
        end
        cls_list = unique(extractfield(gt_all_objects, 'name'));
        
        for kk = 1:length(cls_list)
            
            cls_id = find(strcmp(cls_list{kk}, name_list) == 1);
            curr_gt_bbox = gt_bbox(...
                strcmp(cls_list{kk}, extractfield(gt_all_objects, 'name')), :);
            try
                bbox_candidate = floor(boxes(1:top_k, 1:4));
            catch
                bbox_candidate = floor(boxes(:, 1:4));
            end
            
            [true_overlap, ~] = compute_overlap(curr_gt_bbox, bbox_candidate);
            correct_inst = sum(extractfield(true_overlap, 'max') >= ov);
            
            recall_per_cls(cls_id).correct_inst = ...
                recall_per_cls(cls_id).correct_inst + correct_inst;
            recall_per_cls(cls_id).total_inst = ...
                recall_per_cls(cls_id).total_inst + size(curr_gt_bbox, 1);
        end
    end
end

disp('');
for i = 1:length(cls_dir)
    recall_per_cls(i).recall = ...
        recall_per_cls(i).correct_inst/recall_per_cls(i).total_inst;
    fprintf('cls #%3d: %s\t\trecall: %.4f\n', ...
        i, recall_per_cls(i).name, recall_per_cls(i).recall);
end
mean_recall = mean(extractfield(recall_per_cls, 'recall'));
disp(mean_recall);