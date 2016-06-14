function image_db = load_image_db(varargin)
% load_image_db load the images and annotation data of the asked dataset.
% For example,
% image_db = load_image_db('dataset', 'pascal','set_name','test_2007');
% will return the image_db object that contains the images and the ground
% truth bounding boxes of the PASCAL VOC 2007 test set. 
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
% ************************** OPTIONS *************************************
ip = inputParser;
ip.addParamValue('dataset', 'pascal');
ip.addParamValue('set_name', {})
ip.addParamValue('subset',   {});

voc_path  = [pwd, '/datasets/VOC%s/'];
coco_path = [pwd, '/datasets/MSCOCO/'];

ip.parse(varargin{:});
opts = ip.Results;

subset = opts.subset;
if ~iscell(subset), subset = {subset}; end

set_name = opts.set_name;
if ~iscell(set_name), set_name = {set_name}; end
assert(ischar(set_name{1}));
num_sets = length(set_name);

for i = 1:num_sets
    % add the paths to the images and the grount truth bounding boxes in
    % the image_db structure
    switch opts.dataset
        case 'pascal'
            image_db_this = load_pascal_dataset(set_name{i}, voc_path);
        case 'mscoco'
            image_db_this = load_mscoco_dataset(set_name{i}, coco_path);
    end
    if (~isempty(subset) && ~isempty(subset{i}))
        image_db_this = get_image_subset(image_db_this, subset{i});
        set_name{i}   = [set_name{i}, sprintf('_start%d_stop%d',start_idx, stop_idx)];
    end    
    image_db_all(i) = image_db_this;
end
image_db_all = image_db_all(:);

fnames = fieldnames(image_db_all);
for i = 1:length(fnames)
    image_db.(fnames{i}) = vertcat(image_db_all.(fnames{i})); 
end

switch opts.dataset
    case 'pascal'
        voc_path_year = sprintf(voc_path, '2007');
        VOCopts  = initVOCOpts(voc_path_year,'2007');
        image_db.categories = struct;
        image_db.categories.names      = VOCopts.classes;
        image_db.categories.Ids        = (1:length(VOCopts.classes))';
        image_db.categories.IdsToIndex = (1:length(VOCopts.classes))';
        image_set_name = 'voc_';
        
        if strcmp(set_name,'test_2007')
            save_file_image_names = fullfile(pwd,'sup_data','pascal_all_image_names_test_2007.mat');
            ld = load(save_file_image_names, 'imgIds');
            image_db.imgIds = ld.imgIds(:);
        end
    case 'mscoco'
        save_file_categories  = fullfile(coco_path,'matlab_files','mscoco_categoties.mat');
        ld = load(save_file_categories,'category_names','catIds','catIdsToIndex');
        image_db.categories = struct;
        image_db.categories.names      = ld.category_names;
        image_db.categories.Ids        = ld.catIds;
        image_db.categories.IdsToIndex = ld.catIdsToIndex;  
        image_set_name = 'mscoco_'; 
end

for i = 1:num_sets, image_set_name = [image_set_name, set_name{i}, '_']; end

image_set_name_all = image_set_name(1:end-1);
image_db.image_set_name = image_set_name_all;
end

function image_db = load_pascal_dataset(set_name, voc_path)
image_db = struct;
voc_year = set_name((end-3):end);
img_set  = set_name(1:(end-5));
voc_path_dataset = sprintf(voc_path, voc_year);
assert(exist(voc_path_dataset,'dir')>0);
voc_path_devkit = fullfile(voc_path_dataset,'VOCdevkit');
assert(exist(voc_path_devkit,'dir')>0);

image_db.image_paths = get_image_paths_from_voc(voc_path_devkit, img_set, voc_year);
image_db.all_bbox_gt = get_grount_truth_bboxes_from_voc(voc_path_devkit, img_set, voc_year, true, voc_path_dataset);
image_db.image_sizes = get_img_size(image_db.image_paths);

end

function image_db = load_mscoco_dataset(set_name, data_dir)
image_db = struct;
set_name = regexprep(set_name,'_','');
save_file_bbox_gt     = fullfile(data_dir,'matlab_files', sprintf('mscoco_all_bbox_gt_%s.mat',    set_name));
save_file_image_names = fullfile(data_dir,'matlab_files', sprintf('mscoco_all_image_names_%s.mat',set_name));
ld = load(save_file_image_names, 'image_names', 'image_sizes','imgIds');

image_names          = ld.image_names;
image_db.image_paths = strcat(sprintf('%s/images/%s/', data_dir, set_name), image_names);
image_db.image_sizes = ld.image_sizes;
image_db.imgIds      = ld.imgIds(:);

if (exist(save_file_bbox_gt,'file') > 0)
    ld = load(save_file_bbox_gt);
    image_db.all_bbox_gt    = ld.all_bbox_gt;
    image_db.all_bbox_crowd = ld.all_bbox_crowd;
else
    image_db.all_bbox_gt = cell(length(image_db.image_paths),1);
    for i = 1:length(image_db.all_bbox_gt)
        image_db.all_bbox_gt{i} = zeros(0,6,'single');
    end
end

end

function image_sizes = get_img_size(image_paths)
num_imgs    = numel(image_paths);
image_sizes = zeros(num_imgs,2);

for img_idx = 1:num_imgs
    im_info                = imfinfo(image_paths{img_idx});
    image_sizes(img_idx,1) = im_info.Height;
    image_sizes(img_idx,2) = im_info.Width;
end
end

function image_db = get_image_subset(image_db, subset)
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
    assert(size(data,1)==num_imgs)
    image_db.(fnames{i}) = data(start_idx:stop_idx,:); 
end
end