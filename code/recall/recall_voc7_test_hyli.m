clc; clear; close all;

% revised by Hongyang
% date:         July 7, 2016
% to-do:        a lot

dataset_root = '/media/hongyang/research_at_large/Q-dataset/pascal/VOCdevkit/VOC2007';
annopath = [dataset_root '/Annotations/'];

testset = [dataset_root '/ImageSets/Main/*_test.txt'];
testset_dir = dir(testset);
exclude_hard = true;

% if top_k is not indicated, it will evaluate all proposals
top_k = 300; %2000;
% overlap ratio
ov = 0.5;

%% the provided object proposals
% 'proposals' is a cell-type variable
proposal_path = '../../box_proposals/author_provide/pascal_test_2007/boxes';

%% compute recall
recall_result(length(testset_dir)).testImgNum = 0;

for i = 1:length(testset_dir)
    
    % per class
    cls_name = testset_dir(i).name(1:end-9);    
    fid = fopen([dataset_root '/ImageSets/Main/' testset_dir(i).name], 'r');
    temp = textscan(fid, '%s%s');
    test_gt_ind = cell2mat(cellfun(@(x) str2double(x)==1, temp{2}, 'UniformOutput', false));
    % 'cls_im_ind' is the image index of the current class
    cls_im_ind = find(test_gt_ind == 1);
    test_list = temp{1}(test_gt_ind);
    
    correct_instTotal = 0;
    instTotal = 0;    
    
    % per image
    % compute recall (per instance! not per image)
    % try j = 16 if you debug
    for j = 1:length(test_list)
        
        % first collect GT boxes of this class in this image
        rec = PASreadrecord([annopath, test_list{j}, '.xml']);
        temp = squeeze(struct2cell(rec.objects));
        cls_ind = cellfun(@(x) strcmp(x, cls_name), temp(1,:));
        
        cls_GT_bbox = cell2mat(temp(8, :)');
        cls_GT_bbox = cls_GT_bbox(cls_ind, :);
        if exclude_hard
            difficut_ind = cell2mat(temp(5, :)');
            difficut_ind = difficut_ind(cls_ind);  
            cls_GT_bbox = cls_GT_bbox(~difficut_ind, :);     
        end
        
        try
            bbox_temp = proposals{cls_im_ind(j)};
        catch
            load([proposal_path '/' sprintf('%06d.mat', str2double(test_list{j}))]);
            bbox_temp = boxes;
            clear boxes;
        end
        
        try
            bbox_candidate = floor(bbox_temp(1:top_k, 1:4));
        catch
            bbox_candidate = floor(bbox_temp(:, 1:4));
        end
        
        [true_overlap, ~] = compute_overlap(cls_GT_bbox, bbox_candidate);
        correct_inst = sum(extractfield(true_overlap, 'max') >= ov);     
        correct_instTotal = correct_instTotal + correct_inst;
        instTotal = instTotal + size(cls_GT_bbox, 1);
    end
    
    recall_result(i).cls_name = cls_name;
    recall_result(i).testImgNum = length(test_list);
    recall_result(i).corrInst = correct_instTotal;
    recall_result(i).Inst = instTotal;
    recall_result(i).recall = correct_instTotal/instTotal;
    fprintf('%s,\t\trecall %.3f\n', cls_name, recall_result(i).recall);
end
mean_recall = mean(extractfield(recall_result, 'recall'));
disp('');
disp(mean_recall);