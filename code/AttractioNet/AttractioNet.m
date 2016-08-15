function [ bbox_props_out, bbox_uncut, extra_out_data] = AttractioNet(model, img, conf)
% AttractioNet extractes the box proposals from a single image.
%
% INPUTS:
% (1) model: the AttractioNet model in the form of a matlab struct data
% type.
% (2) img: the input RGB image in the form of H x W x 3 uint8 array.
% (3) conf: struct with the AttractioNet configuration options.
%
% OUTPUTS:
% (1) bbox_props_out: is a K x 5 single precision array where K is the
% number of output box proposals (by default K=2000).
% (2) extra_out_data: struct that contains intermediate data of the box
% proposal extraction process.

model.max_rois_num_in_gpu = 1000;

% TODO: pre-caution on test15: if the image is too small, just skip it.

%************************* GENERATE SEED BOXES ****************************
% generate seed boxes uniformly distributed arround the image
bbox_props_in = single(AttractioNet_create_seed_boxes(img, conf.num_seed_boxes));
%**************************************************************************

%************************* ATTEND REFINE REPEAT ***************************
conf.threshold = -Inf;
bbox_props_cands_per_iter = cell(conf.num_iterations,1);
skip_image_conv_layers = false;

for iter = 1:conf.num_iterations
    
    % APPLY THE CATEGORY AGNOSTIC OBJECTNESS SCORING MODULE:
    bboxes_scores  = AttractioNet_objectness_scoring(model, img, bbox_props_in, skip_image_conv_layers);
    
    if (iter == 1)
        % Only for the first iteration
        [bbox_props_in, bboxes_scores] = reduce_box_proposals_num(bbox_props_in, bboxes_scores, [], conf);
        skip_image_conv_layers = true;
    end
    
    % APPLY THE CATEGORY AGNOSTIC OBJECT LOCATION REFINEMENT MODULE:
    bbox_refined = AttractioNet_object_location_refinemt(model, img, bbox_props_in, skip_image_conv_layers);
    
    % refined box coordinates and objectness scores
    bbox_props_cands_this = single([bbox_refined(:,1:4), bboxes_scores]);
    % store the candidate box proposals of this iteration
    bbox_props_cands_per_iter{iter} = bbox_props_cands_this;
    
    %************ PREPARE THE INPUT BOXES FOR THE NEXT ITERATION **********
    if (iter < conf.num_iterations)
        bbox_props_in = reduce_box_proposals_num(bbox_refined, bboxes_scores, bbox_props_in, conf);
    end
    %**********************************************************************
end
% Merge the candidate box proposals actively generated during all iterations
bbox_props_cands = cell2mat(bbox_props_cands_per_iter);
bbox_uncut = bbox_props_cands;
% save('bbox_props_cands_Aug_1_default.mat', 'bbox_props_cands');
%*************************** POST-PROCESSING ******************************
% Apply the final step of multi-threshold non-max-suppression that returns
% final list of box proposals.
bbox_props_out = AttractioNet_postprocess(bbox_props_cands, ...
    'thresholds',       conf.threshold, ...
    'use_gpu',          true, ...
    'mult_thr_nms',     length(conf.nms_iou_thrs)>1, ...
    'nms_iou_thrs',     conf.nms_iou_thrs, ...
    'max_per_image',    conf.max_per_image);

extra_out_data.bbox_props_cands = bbox_props_cands;
extra_out_data.bbox_props_cands_per_iter = bbox_props_cands_per_iter;

if conf.multiple_nms_test
    proposals = cell(1+length(conf.nms_range), 1);
    proposals{1} = bbox_props_out;
    
    for i = 1:length(conf.nms_range)
        proposals{1+i} = AttractioNet_postprocess(bbox_props_cands, ...
            'nms_iou_thrs',     conf.nms_range(i), ...
            'max_per_image',    2000);
    end
    bbox_props_out = proposals;
end
%**************************************************************************
end

function [bboxes_coord, bboxes_scores_descend] = ...
    reduce_box_proposals_num(bboxes_coord, bboxes_scores, bboxes_coord_prev, conf)
% It reduces the candidate box proposal by performing the following operations:
% (1) (optional) early stop sequences of bounding box predictions that have
%      already converged. A sequence is considered to have converged if the
%      IoU of the previous input box (bboxes_coord_prev) with the predicted
%      box (bboxes_coord) is greater than conf.iou_thrs_close. This step is
%      optional and depends if the previous boxes (bboxes_coord_prev) is given
% (2) Apply non-maximum-suppresion of the very close candidate box proposals
%     with IoU threshold of conf.iou_thrs_close (typical value: 0.9)
% (3) Keep the top conf.max_per_image_init (=2000) candidate box proposals


if ~isempty(bboxes_coord_prev)
    is_not_converged     = boxoverlap(bboxes_coord_prev(:,1:4), bboxes_coord(:,1:4), true) < conf.iou_thrs_close;
    scored_box_proposals = single([bboxes_coord(is_not_converged,1:4), bboxes_scores(is_not_converged,:)]);
else
    scored_box_proposals = single([bboxes_coord(:,1:4), bboxes_scores]);
end
% AttractioNet_postprocess(), in this case applied a simple
% non-maximum-suppresion step with IoU threshold equal to conf.iou_thrs_close
% and then keeps the top conf.max_per_image_init bounding boxes
scored_box_proposals = AttractioNet_postprocess(scored_box_proposals, ...
    'thresholds', conf.threshold, 'nms_iou_thrs', conf.iou_thrs_close, ...
    'max_per_image', conf.max_per_image_init, 'use_gpu', true);
bboxes_coord  = scored_box_proposals(:,1:4);
bboxes_scores_descend = scored_box_proposals(:,5);
end