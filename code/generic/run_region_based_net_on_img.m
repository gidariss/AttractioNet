function [outputs, output_blob_names] = run_region_based_net_on_img(...
    model, image, bboxes, varargin)
% run_region_based_net_on_img: applies on the candidate bounding boxes
% (bboxes) and on the image the provided region-based CNN network (model).
% 
% INPUTS:
% 1) model:  (type struct) the bounding box recognition model
% 2) image:  a [Height x Width x 3] uint8 matrix with the image 
% 3) bboxes: a N x 4 array with the bounding box coordinates
% 4) out_blob_names_extra: (optional) a cell array with network blob names 
% (in the form of strings) of which the data will be returned from the 
% function. By default the function will always return the data of the
% network output blobs.
% 5) skip_image_conv_layers: (optional) If set to true it skips forwarding
% the image from the first convolutional layers of the CNN and it starts 
% directly from forwarding the regions throught the network. Default value: false
%
% OUTPUTS:
% 1) outputs: a NB x 1 cell array with the data of the inquired blobs; the
% i-th element outputs{i} has the data of the network blob with name 
% out_blob_names_total{i}.
% 2) out_blob_names_total: a NB x 1 cell array with the inquired blob name
% (in the form of strings)

ip = inputParser;
ip.addParamValue('input_blob_names',    model.net.inputs,  @iscell);
ip.addParamValue('output_blob_names',   model.net.outputs, @iscell);
ip.addParamValue('skip_image_conv_layers', false, @islogical);
ip.parse(varargin{:});
opts = ip.Results;

if isempty(opts.input_blob_names),  opts.input_blob_names  = model.net.inputs; end
if isempty(opts.output_blob_names), opts.output_blob_names = model.net.outputs; end

% max_rois_num_in_gpu: maximum number of regions that will be given in one
% go in the network such that they could fit in the GPU memory
max_rois_num_in_gpu = model.max_rois_num_in_gpu; 
% out_blob_names_total = [out_blob_names(:); out_blob_names_extra(:)];
output_blob_names = opts.output_blob_names;  % name of the network's output blobs
input_blob_names  = opts.input_blob_names; % name of the network's input blobs

image_size = size(image);
% get the image blob(s) that will be fed as input to the caffe network
[im_blob, im_scales] = get_image_blob(model, image);

% get the region blobs that will be given as input to caffe network
[rois_blob, scale_ids] = map_boxes_to_regions_wrapper(...
    model.pooler, bboxes, image_size, im_scales);
% divide the regions in chunks of maximum size of max_rois_num_in_gpu
num_rois   = numel(scale_ids{1});
num_chunks = ceil(num_rois / max_rois_num_in_gpu);

% fed the image blobs and on the network and get the output
outputs = run_region_based_net(model, ...
    output_blob_names, input_blob_names, im_blob, rois_blob, ...
    num_rois, num_chunks, max_rois_num_in_gpu, opts.skip_image_conv_layers);    
outputs = cellfun(@(x) reshape(x,[1, 1, 1, num_chunks]), outputs, 'UniformOutput', false);

% format appropriately the output per blob
outputs = format_outputs(outputs, num_rois, num_chunks);
end

function [im_blob, im_scales] = get_image_blob(model, image)
[im_blob, im_scales] = arrayfun(@(x) ...
    prepare_img_blob(image, x, model.mean_pix, model.max_size), ...
    model.scales, 'UniformOutput', false);    
im_scales = cell2mat(im_scales);
if length(model.scales) > 1
    im_blob = {img_list_to_blob(im_blob)};
end
end  

function caffe_reshape_net_as_this_input(net, inputs, num_rois)
input_size = cellfun(@(x) size(x), inputs, 'UniformOutput', false);

% a very stupid fix.
if num_rois == 1
    for i = 2:length(input_size)
        input_size{i} = [input_size{i}, 1];
    end
end

input_blob_names = net.inputs;
for i = 1:length(input_size)
    size_this = net.blobs(input_blob_names{i}).shape();
    input_size_this = ones(size(size_this));
    input_size_this(1:length(input_size{i})) = input_size{i};
    input_size{i} = input_size_this;
end

caffe_reshape_net(net, input_size);
end

function [rois_blob, scale_ids] = map_boxes_to_regions_wrapper(...
    region_params, bboxes, image_size, im_scales)

num_region_types = length(region_params);
rois_blob = cell(1,num_region_types);
scale_ids = cell(1,num_region_types);
% analyze each bounding box to one or more type of regions
for r = 1:num_region_types 
    rois_blob{r} = map_boxes_to_regions_mult_scales(...
        region_params(r), bboxes, image_size, im_scales);   
    scale_ids{r} = rois_blob{r}(:,1);
end
for r = 1:num_region_types
    rois_blob{r} = rois_blob{r} - 1; % to c's index (start from 0)
    rois_blob{r} = single(permute(rois_blob{r}, [3, 4, 2, 1]));    
end
end

function outputs = run_region_based_net(model, ...
    out_blob_names, input_blob_names_selected, im_blob, rois_blob, ...
    num_rois, num_chunks, max_rois_num_in_gpu, skip_image_conv_layers)

outputs = cell(length(out_blob_names),1);
outputs = cellfun(@(x) cell([1, 1, 1, num_chunks]), outputs, 'UniformOutput', false);
for i = 1:num_chunks
    sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
    sub_ind_end   = min(num_rois, i * max_rois_num_in_gpu);
    sub_num_rois  = sub_ind_end-sub_ind_start+1;
    sub_rois_blob = slice_rois_blob(rois_blob, sub_ind_start:sub_ind_end, num_rois);
    
    % the inputs to the network for this chunk
    net_inputs = [im_blob, sub_rois_blob];
    caffe_reshape_net_as_this_input(model.net, net_inputs, sub_num_rois);
    input_blob_names_all = model.net.inputs; % list with names of all the input blobs in the net
    % remove from net_inputs input blobs that are not in the list: input_blob_names_selected
    net_inputs = remove_input_blobs_not_in_the_list(input_blob_names_all, input_blob_names_selected, net_inputs);
    input_blob_names = input_blob_names_selected;

    if (i > 1 || skip_image_conv_layers)
        % skip extracting the convolutional features of the same image again.
        input_blob_names = input_blob_names(2:end); % remove the image blob from the input blob names list
        net_inputs = net_inputs(2:end);  % remove the image blob from the  input blobs list        
    end
    caffe_set_blobs_data(model.net, net_inputs, input_blob_names);
    startLayerIdx = get_start_layer_idx_of_blobs(model.net, input_blob_names);
    stopLayerIdx  = get_stop_layer_idx_of_blobs( model.net, out_blob_names);
    model.net.forward_prefilled_from_to(startLayerIdx, stopLayerIdx);
    for j = 1:length(out_blob_names)
        outputs{j}{i} = model.net.blobs(out_blob_names{j}).get_data();           
    end
end 
end

function sub_rois_blobs = slice_rois_blob(rois_blobs, slice_inds, num_rois)
sub_rois_blobs = cell(1,length(rois_blobs)); % the regions of this chunk that will be fed to the network
added_number = single((num_rois==1));
for r = 1:length(rois_blobs)
    switch (ndims(rois_blobs{r}) + added_number)
        case 1
            sub_rois_blobs{r} = rois_blobs{r}(slice_inds);
        case 2
            sub_rois_blobs{r} = rois_blobs{r}(:,slice_inds);
        case 3
            sub_rois_blobs{r} = rois_blobs{r}(:,:,slice_inds);
        case 4
            sub_rois_blobs{r} = rois_blobs{r}(:,:,:,slice_inds);
        otherwise
            error('not supported tensor size; num dims = %d', ndims(sub_rois_blobs{r}))
    end
end
end

function outputs = format_outputs(outputs, num_rois, num_chunks)
% properly format the outputs

num_out_blobs_total = length(outputs);
for j = 1:num_out_blobs_total
    if num_chunks == 1
        outputs{j} = squeeze(cell2mat(outputs{j}));
    else
        chunk_sizes = cell2mat(cellfun(@(x) ...
            [size(x,1), size(x,2), size(x,3), size(x,4)], ...
            squeeze(outputs{j}(:)), 'UniformOutput', false));
        dim = find(any(chunk_sizes ~= 1,1),1,'last');
        if (num_rois <= 1), dim = dim + 1; end
        
        shape = ones([1,dim]);
        shape(end)=num_chunks;
        outputs{j} = squeeze(cell2mat(reshape(outputs{j}, shape)));
    end
end
end

function [input_blobs] = remove_input_blobs_not_in_the_list(...
    input_blob_names_all, input_blob_names_selected, input_blobs)
keep_blobs_mask = false(numel(input_blob_names_all),1);
for i = 1:numel(input_blob_names_all)
    if any(strcmp(input_blob_names_all{i}, input_blob_names_selected))
        keep_blobs_mask(i) = true; % if in the list input_blob_names_selected then keep it.
    end
end
input_blobs = input_blobs(keep_blobs_mask);
end

function startLayerIdx = get_start_layer_idx_of_blobs(net, blob_names)
num_blobs = length(blob_names);
min_layer_id_per_blob = zeros(num_blobs,1);
for i = 1:num_blobs
    layer_ids = net.layer_ids_with_input_blob(blob_names{i});
    assert(~isempty(layer_ids));
    min_layer_id_per_blob(i) = min(layer_ids);
end
startLayerIdx = min(min_layer_id_per_blob);
end

function stopLayerIdx = get_stop_layer_idx_of_blobs(net, blob_names)
num_blobs = length(blob_names);
layer_id_per_blob = zeros(num_blobs,1);
for i = 1:num_blobs
    layer_id = net.layer_id_with_output_blob(blob_names{i});
    assert(numel(layer_id) == 1);
    layer_id_per_blob(i) = layer_id;
end
stopLayerIdx = max(layer_id_per_blob);
end