function AttractioNet_extract_box_proposals_from_dataset(model_dir_name, varargin)
% AttractioNet_extract_box_proposals_from_dataset: extracts the
% AttractioNet bounding box proposals from a dataset of images.
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

%****************************** OPTIONS ***********************************
ip = inputParser;
ip.addParamValue('gpu_id', 0, @isscalar); % the gpu that will be used for extracting the proposals (1-based).
ip.addParamValue('set_name','val_2014', @ischar); % image set name, e.g. val_2014, test_2007, ...
ip.addParamValue('dataset', 'mscoco',   @ischar); % dataset name, e.g. mscoco, pascal.
ip.addParamValue('startIdx',   [], @isnumeric); % start image index
ip.addParamValue('stopIdx',    [], @isnumeric); % end image index
ip.addParamValue('first_5k_coco_val_2014', false,  @islogical);
ip.addParamValue('eval_props',  false,  @islogical); % evaluate the extracted bounding box proposals
ip.addParamValue('suffix', '', @ischar); 
ip.parse(varargin{:});
opts = ip.Results;
gpu_id = opts.gpu_id;
%**************************************************************************
%******************************* LOAD DATASET *****************************
if opts.first_5k_coco_val_2014
    opts.dataset = 'mscoco';
    opts.set_name  = 'val_2014';
    opts.startIdx  = [];
    opts.stopIdx   = [];
end
image_db = load_image_db('dataset', opts.dataset,'set_name', opts.set_name); 
image_db = get_image_subset(image_db, [opts.startIdx, opts.stopIdx]);
if opts.first_5k_coco_val_2014
    image_db = get_first_5k_coco_val2014_images(image_db);
end
fprintf('Number of images: %d\n',length(image_db.image_paths));
image_paths = image_db.image_paths;
all_bbox_gt = image_db.all_bbox_gt;
%**************************************************************************
%************************* SET GPU/CPU DEVICE *****************************
caffe_set_device( gpu_id );
caffe.reset_all();
%**************************************************************************
%***************************** LOAD MODEL *********************************
full_model_dir  = fullfile(pwd, 'models-exps', model_dir_name); 
mat_file_name = 'box_proposal_model.mat';
model = AttractioNet_load_model(full_model_dir, mat_file_name);
%**************************************************************************
%********************** CONFIGURATION PARAMETERS **************************
box_prop_conf = AttractioNet_get_defaul_conf();
fprintf('AttractioNet configuration params:\n');
disp(box_prop_conf);
model.scales   = box_prop_conf.scales;   
model.max_size = box_prop_conf.max_size; 
fprintf('AttractioNet Model:\n');
disp(model);
%**************************************************************************
%******************* SET BOX PROPOSALS DESTINATION DIR ********************
dst_model_dir   = fullfile(pwd, 'box_proposals', model_dir_name); % Father directory named after the model directory name.
dst_method_dir  = fullfile(dst_model_dir, 'AttractioNet_Boxes');  % Directory where all the box proposal files (one per image) will be saved.
dataset_suffix  = sprintf('%s%s',image_db.image_set_name,opts.suffix); 
dst_dataset_dir = fullfile(dst_method_dir, dataset_suffix); 
% create directories
mkdir_if_missing(dst_model_dir); 
mkdir_if_missing(dst_method_dir);
mkdir_if_missing(dst_dataset_dir);
fprintf('Destination directory tree:\n')
fprintf('\tmodel   dir: %s\n',  dst_model_dir);
fprintf('\tmethod  dir: %s\n',  dst_method_dir);
fprintf('\tdataset dir: %s\n',  dst_dataset_dir);
%**************************************************************************
%*********** EXTRACT BOUNDING BOX PROPOSALS FROM SET OF IMAGES ************
% Note that the following command will extract the bounding box proposals 
% of the given images and save the them (using a single matlab file per 
% image) on the directory [dst_dataset_dir,'/boxes/'].
[all_bboxes, all_bboxes_filepaths] = AttractioNet_run_on_dataset(model, ...
    image_paths, dst_dataset_dir, 'conf', box_prop_conf);
% all_bboxes: is a N x 1 cell array, where N is number of images, with the
% box proposals of each image. Specifically, all_bboxes{i} is a K x 5 single 
% precision numeric array with the box proposals of the i-th image (where 
% K is the number of box proposals; by default 2000)
% all_bboxes_filepaths: is a N x 1 cell array with the paths to the matlab
% files that contain the extracted box proposals. Specifically,
% all_bboxes_filepaths{i} is a string of the path to the matlab file that
% contains the box proposals of the i-th image.
%**************************************************************************
%********************* FREE GPU/CPU MEMORY (Caffe) ************************
caffe.reset_all();
%**************************************************************************
%********************* EVALUATE BOUNDING BOX PROPOSALS ********************
if opts.eval_props
    if (strcmp(opts.dataset,'pascal') && strcmp(opts.set_name,'test_2007')) || ...
       (strcmp(opts.dataset,'mscoco') && (strcmp(opts.set_name,'train_2014') || strcmp(opts.set_name,'val_2014')))
        resFile_json_file = fullfile(dst_dataset_dir, sprintf('boxes_%s.json',image_db.image_set_name));
        prepareCOCOStyleBoxPropResults(image_db, {all_bboxes}, resFile_json_file);
        evaluateProposalWithCOCO_API(opts.dataset, opts.set_name, image_db.imgIds, resFile_json_file)
    end  
end
%**************************************************************************
end

function image_db = get_image_subset(image_db, subset)

if ~isempty(subset)
    assert(isnumeric(subset) && numel(subset)==2);
    start_idx = subset(1); 
    stop_idx  = subset(2);
    num_imgs  = length(image_db.image_paths);

    start_idx = max(1, start_idx);
    stop_idx  = min(num_imgs, stop_idx);
    if stop_idx < 0, stop_idx = num_imgs; end

    fnames = fieldnames(image_db);
    for i = 1:length(fnames)
        data = image_db.(fnames{i});
        if (isnumeric(data) || iscell(data)) && (size(data,1) == num_imgs)
            image_db.(fnames{i}) = data(start_idx:stop_idx,:); 
        end
    end
end

end
function image_db = get_first_5k_coco_val2014_images(image_db)

save_file_first5k_imgIds = fullfile(pwd,'sup_data','first5k_imgIds_of_coco_val2014.mat');
assert(exist(save_file_first5k_imgIds,'file')>0);
ld = load(save_file_first5k_imgIds, 'first5K_image_ids');    
first5K_image_ids = ld.first5K_image_ids; clear ld;

[~,inds] = intersect(image_db.imgIds,first5K_image_ids,'rows');
assert(length(inds)  == length(first5K_image_ids));

num_imgs = length(image_db.imgIds);
fnames   = fieldnames(image_db);
for i = 1:length(fnames)
    data = image_db.(fnames{i});
    if (isnumeric(data) || iscell(data)) && (size(data,1) == num_imgs)
        image_db.(fnames{i}) = data(inds,:); 
    end
end
end