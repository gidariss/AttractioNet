function scores = AttractioNet_objectness_scoring(model, image, bboxes, skip_image_conv_layers)
% AttractioNet_objectness_scoring:
% It implements the objectness scoring module of AttractioNet. The
% objectness scoring module assign confidence score to each input box that
% represents how likely it is each box to tightly enclose an object (of any
% category).
%
% INPUTS:
% 1) model:  (data type struct) the AttractioNet model
% 2) image:  a [Height x Width x 3] uint8 matrix with the image 
% 3) bboxes: a N x 4 array with the bounding box coordinates; each row is 
% the oordinates of a bounding box in the form of [x0,y0,x1,y1] where 
% (x0,y0) is tot-left corner and (x1,y1) is the bottom-right corner. N is 
% the number of bounding boxes.
% 4) skip_image_conv_layers: boolean value; if true it skips extracting the
% image convolutional feature maps (in this cases it is assumed that the
% image convolutional feature maps have already been extracted and are 
% stored in the caffe buffers 
%
% OUTPUT:
% scores: N x 1 array with the objectness scores of each bounding box. N 
% is the number of bounding boxes.
% 
% This file is part of the code that implements the following paper:
% Title      : "Attend Refine Repeat: Active Box Proposal Generation via In-Out Localization"
% Authors    : Spyros Gidaris, Nikos Komodakis
% Institution: Universite Paris Est, Ecole des Ponts ParisTech
% code       : https://github.com/gidariss/AttractioNet
%
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2016 Spyros Gidaris
%
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

if ~exist('skip_image_conv_layers','var'), skip_image_conv_layers = false; end

num_classes = length(model.classes); % it should be 1 for the objectness category
assert(num_classes==1);
if isempty(bboxes)
    scores = zeros(0,num_classes,'single');
    return;
end

% names of the input data (blobs using the caffe notation) of this (sub-)network; 
% for example, in the case of the objectness scoring module those ae the
% entire image and the input bounding box coordinates
input_blob_names_ = {}; 
if isfield(model,'recognition_input_blob_names')
    input_blob_names_ = model.recognition_input_blob_names;
end
% name(s) of the output data (blobs using the caffe notation) of this
% (sub-)network. In the case of the objectness scoring module those are the
% objectness scores
output_blob_names_ = model.score_out_blob;
if ~iscell(input_blob_names_),  input_blob_names_  = {input_blob_names_}; end
if ~iscell(output_blob_names_), output_blob_names_ = {output_blob_names_}; end

% apply on the candidate bounding boxes and on the image the region-based 
% CNN (sub-)network that implements the objectness scoring module
[outputs, output_blob_names] = run_region_based_net_on_img(model, image, bboxes, ...
    'output_blob_names', output_blob_names_, 'input_blob_names', input_blob_names_, ...
    'skip_image_conv_layers',skip_image_conv_layers);

% get the output blob that corresponds to the confidence scores of the
% bounding boxes
idx = find(strcmp(output_blob_names,model.score_out_blob));
assert(numel(idx) == 1);
scores = outputs{idx}';

% in case that in the scores array there is an exra column than the number 
% of categories, then the first column represents the confidence score of 
% each bounding box to be on background and it is removed before the score 
% array is returned.

% If there are more than 1 outputs - probabilities per input box, then the
% first probability is the background probability and the second the 
% objectness probability. In this case remove the column that correspond to
% the background probabilities and keep only the objectness probabilities
if size(scores,2) == (num_classes + 1), scores = scores(:,2:end); end
end