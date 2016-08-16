function bboxes_out = AttractioNet_object_location_refinemt(model, image, bboxes_in, skip_image_conv_layers)
% AttractioNet_object_location_refinemt:
% It implements the object location refinement module of AttractioNet.
% Specifically, given the input boxes it predicts a new location for each 
% box such that the new ones will be closer (i.e. better localized) on the 
% objects (regadless of their category) that are closest to the input boxes
%
% INPUTS:
% 1) model:  (data type struct) the AttractioNet model
% 2) image:  a [Height x Width x 3] uint8 matrix with the image 
% 3) bboxes_in: a N x 4 array with the bounding box coordinates
%
% OUTPUT:
% 1) bboxes_out : a N x 4 array with the refined bounding boxes. It has the
% same format as bboxes_in

if ~exist('skip_image_conv_layers','var'), skip_image_conv_layers = false; end

if isempty(bboxes_in)
    bboxes_out = zeros(0,5,'single');
    return;
end

% names of the input data (blobs using the caffe notation) of this (sub-)network.
% For example, in the case of the object location refinement module those
% are the entire image and the input bounding box coordinates
input_blob_names_ = {};
if isfield(model,'localization_input_blob_names')
    input_blob_names_ = model.localization_input_blob_names;
end
% name(s) of the output data (blobs using the caffe notation) of this
% (sub-)network. In the case of the object location refinement module those 
% are the in-out probability vectors (computed for each input box)
output_blob_names_ = model.preds_loc_out_blob;
if ~iscell(input_blob_names_),  input_blob_names_  = {input_blob_names_}; end
if ~iscell(output_blob_names_), output_blob_names_ = {output_blob_names_}; end

% apply on the candidate bounding boxes and on the image the region-based 
% CNN (sub-)network that implements the objectness scoring module
[outputs, output_blob_names] = run_region_based_net_on_img(model, image, bboxes_in(:,1:4), ...
    'output_blob_names', output_blob_names_, 'input_blob_names', input_blob_names_, ...
    'skip_image_conv_layers', skip_image_conv_layers);
% get the output blob that corresponds on the predicted in-out probability vectors
idx = find(strcmp(output_blob_names,model.preds_loc_out_blob));
assert(numel(idx) == 1);
location_probability_vectors = outputs{idx};

% Decode the location probability vectors (in-out probability vectors) in
% order to get the refined box coordinates
img_size = [size(image,1), size(image,2)];
bboxes_out = decode_location_probability_vectors_to_bbox_preds(bboxes_in(:,1:4), ...
    ones([size(bboxes_in,1),1],'single'), location_probability_vectors, model.loc_params); 

bboxes_out(:,1:4) = clip_bbox_inside_the_img(bboxes_out(:,1:4), img_size);
bboxes_out(:,1:4) = check_box_coords(bboxes_out(:,1:4));
end

function bboxes = clip_bbox_inside_the_img(bboxes, img_size)
bboxes(:,1:4:end) = max(1,           bboxes(:,1:4:end));
bboxes(:,2:4:end) = max(1,           bboxes(:,2:4:end));
bboxes(:,3:4:end) = min(img_size(2), bboxes(:,3:4:end));
bboxes(:,4:4:end) = min(img_size(1), bboxes(:,4:4:end));
end

function bboxes = check_box_coords(bboxes)
assert(size(bboxes,2) == 4);
ind = bboxes(:,1) > bboxes(:,3);
if any(ind), bboxes(ind,[3,1]) = bboxes(ind,[1,3]); end
ind = bboxes(:,2) > bboxes(:,4);
if any(ind), bboxes(ind,[4,2]) = bboxes(ind,[2,4]); end
end