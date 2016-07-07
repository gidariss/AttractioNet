function [ bbox_props_out, extra_out_data] = AttractioNet(model, img, conf)
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
%************************* GENERATE SEED BOXES ****************************
% generate seed boxes uniformly distributed arround the image
bbox_props_in = single(AttractioNet_create_seed_boxes(img, conf.num_seed_boxes)); 
%**************************************************************************

%************************* ATTEND REFINE REPEAT ***************************
conf.threshold = -Inf;
bbox_props_cands_per_iter = cell(conf.num_iterations,1);
skip_image_conv_layers = false;
for iter = 1:conf.num_iterations
    %*********************** ATTEND & REFINE PROCEDURE ********************
    % APPLY THE CATEGORY AGNOSTIC OBJECTNESS SCORING MODULE:
    bboxes_scores  = AttractioNet_objectness_scoring(model, img, bbox_props_in, skip_image_conv_layers);
    if (iter == 1) % Only for the first iteration
        % Reduce the number of bounding boxes processed from the subsequent
        % processing steps. it is only necessary for reducing the total runtime
        [bbox_props_in, bboxes_scores] = reduce_box_proposals_num(bbox_props_in, bboxes_scores, [], conf);

        % The first time that the AttractioNet_objectness_scoring() function 
        % is called, the entire image convolutional feature maps are
        % extracted. For the subsequent calls of the functions-modules:
        % 1) AttractioNet_objectness_scoring()
        % 2) AttractioNet_object_location_refinemt()
        % the extraction of the image convolutional feature maps is
        % skipped. Thus the flag skip_image_conv_layers is set to true. 
        skip_image_conv_layers = true;          
    end
    % APPLY THE CATEGORY AGNOSTIC OBJECT LOCATION REFINEMENT MODULE:
    bbox_refined = AttractioNet_object_location_refinemt(model, img, bbox_props_in, skip_image_conv_layers);
    bbox_props_cands_this = single([bbox_refined(:,1:4), bboxes_scores]); % refined box coordinates and objectness scores
    bbox_props_cands_per_iter{iter} = bbox_props_cands_this; % store the candidate box proposals of this iteration
    %**********************************************************************
    
    %************ PREPARE THE INPUT BOXES FOR THE NEXT ITERATION **********
    if (iter < conf.num_iterations)
        % reduce the number of bounding boxes that will be processed from
        % the next iteration; it is only necessary for reducing the runtime
        bbox_props_in = reduce_box_proposals_num(bbox_refined, bboxes_scores, bbox_props_in, conf);
        % bbox_props_in are the input boxes that will be used in the next
        % iteration; note that they actually are a subset of the bbox_refined 
        % that are given as input to the reduce_box_proposals_num() function
    end
    %**********************************************************************
end
% Merge the candidate box proposals actively generated during all the
% iterations
bbox_props_cands = cell2mat(bbox_props_cands_per_iter);
%**************************************************************************

%*************************** POST-PROCESSING ******************************
% Apply the final step of multi-threshold non-max-suppression that returns
% final list of box proposals.
bbox_props_out = AttractioNet_postprocess(bbox_props_cands, ...
    'use_gpu',true, 'thresholds', conf.threshold, 'mult_thr_nms',length(conf.nms_iou_thrs)>1, ...
    'nms_iou_thrs',  conf.nms_iou_thrs, 'max_per_image', conf.max_per_image);
extra_out_data.bbox_props_cands = bbox_props_cands;
extra_out_data.bbox_props_cands_per_iter = bbox_props_cands_per_iter;
%**************************************************************************
end

function [bboxes_coord, bboxes_scores] = reduce_box_proposals_num(bboxes_coord, bboxes_scores, bboxes_coord_prev, conf)
% It reduces the candidate box proposal by performing the following operations:
% (1) (optional) ealy stop sequences of bounding box predictions that have 
%      already converged. A sequence is considered to have converged if the 
%      IoU of the previous input box (bboxes_coord_prev) with the predicted  
%      box (bboxes_coord) is greater than conf.iou_thrs_close. This step is
%      optional and depends if the previous boxes (bboxes_coord_prev) is given
% (2) Apply non-maximum-suppresion of the very close candidate box proposals
%     with IoU threshold of conf.iou_thrs_close (typical value: 0.9)
% (3) Keep the top conf.max_per_image_init (=2000) candidate box proposals
%
% The typical values of conf.iou_thrs_close and conf.max_per_image_init
% that are actually used in AttractioNet are:
% conf.iou_thrs_close = 0.9 
% conf.max_per_image_init = 2000

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
bboxes_scores = scored_box_proposals(:,5);
end