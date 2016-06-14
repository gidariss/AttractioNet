function script_prepare_COCO_matlab_data_files

dataDir        = './datasets/MSCOCO'; 
dst_directory  = sprintf('%s/matlab_files/',dataDir); mkdir_if_missing(dst_directory);
dataTypes_List = {'train2014','val2014'};

fprintf('Prepare MATLAB data files for the COCO dataset:\n')

for d = 1:length(dataTypes_List)
    dataType = dataTypes_List{d};
    
    fprintf('Set %s: ',dataType)

    annotation_file  = sprintf('%s/annotations/instances_%s.json',dataDir,dataType);
    images_directory = sprintf('%s/images/%s/',dataDir,dataType);
    dst_bbox_gt_file = fullfile(dst_directory, sprintf('mscoco_all_bbox_gt_%s.mat', dataType));
    dst_images_file  = fullfile(dst_directory, sprintf('mscoco_all_image_names_%s.mat', dataType));
    
    if (~exist(dst_images_file,'file') || ~exist(dst_bbox_gt_file,'file'))
        prepare_img_and_ann_data_in_matlab_files(images_directory, annotation_file, dst_bbox_gt_file, dst_images_file);
    end
    
    if strcmp(dataType,'val2014')
        dst_cat_data_file = fullfile(dst_directory, 'mscoco_categoties.mat');
        if ~exist(dst_cat_data_file,'file')
            prepare_coco_categories_data(annotation_file, dst_cat_data_file);
        end
    end
    fprintf('\n');
end

dataTypes_List = {'test2015','test-dev2015'};
for d = 1:length(dataTypes_List)
    dataType = dataTypes_List{d};
    fprintf('Set %s: ',dataType)
    img_info_file    = sprintf('%s/annotations/image_info_%s.json',dataDir,dataType);
    images_directory = sprintf('%s/images/%s/',dataDir,dataType);
    dst_images_file  = fullfile(dst_directory, sprintf('mscoco_all_image_names_%s.mat', dataType));
    if ~exist(dst_images_file,'file')
        prepare_test_set_img_data_in_matlab_file(images_directory, img_info_file, dst_images_file);
    end
    fprintf('\n');
end

end

function prepare_img_and_ann_data_in_matlab_files(images_directory, annotation_file, dst_bbox_gt_file, dst_images_file)
coco=CocoApi(annotation_file); 
imgIds = coco.getImgIds();

cats = coco.loadCats(coco.getCatIds());
catIds = coco.getCatIds('catNms',{cats.name});
catIdsToIndex = zeros(max(catIds),1);
catIdsToIndex(catIds) = 1:length(catIds);

img_objs     = loadImgs(coco, imgIds);
image_names  = {img_objs.file_name};
image_sizes  = [img_objs.height; img_objs.width]';
image_names  = image_names(:);
image_paths  = strcat(images_directory, image_names);

num_imgs       = length(image_paths);
all_bbox_gt    = cell( num_imgs, 1);
all_bbox_crowd = cell( num_imgs, 1);
all_obj_area   = cell( num_imgs, 1);

for i = 1:num_imgs
    assert(exist(image_paths{i},'file')>0);
    imgId = imgIds(i);
    annIds = coco.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[]);
    if ~isempty(annIds)
        anns       = coco.loadAnns(annIds); 
        bbox_gt    = cell2mat({anns.bbox}');
        bbox_gt    = single([bbox_gt(:,1:2),bbox_gt(:,3:4)+bbox_gt(:,1:2)-1]);
        area_gt    = single(cell2mat({anns.area}'));
        iscrowd    = single(cell2mat({anns.iscrowd}'));
        
        cat_ids    = [anns.category_id]';
        cat_index  = catIdsToIndex(cat_ids);
        bbox_gt    = single([bbox_gt, cat_index, iscrowd]);
        bbox_crowd = bbox_gt(iscrowd==1,:);
        bbox_gt    = bbox_gt(iscrowd==0,:);
        area_gt    = area_gt(iscrowd==0);
    else
        bbox_gt    = zeros(0,6,'single');
        area_gt    = zeros(0,1,'single');
        bbox_crowd = zeros(0,6,'single');
    end
    all_bbox_gt{i}    = bbox_gt;
    all_bbox_crowd{i} = bbox_crowd;
    all_obj_area{i}   = area_gt;
end

save(dst_bbox_gt_file, 'all_bbox_gt','all_bbox_crowd','all_obj_area','-v7.3');
save(dst_images_file,  'image_names', 'image_sizes', 'imgIds', '-v7.3');

clear coco;
end

function prepare_coco_categories_data(annotation_file, dst_cat_data_file)
coco=CocoApi(annotation_file); 
cats = coco.loadCats(coco.getCatIds());
category_names ={cats.name}; fprintf('COCO categories: ');
fprintf('%s, ',category_names{:}); fprintf('\n');
super_category_names=unique({cats.supercategory}); fprintf('COCO supercategories: ');
fprintf('%s, ',super_category_names{:}); fprintf('\n');

catIds = coco.getCatIds('catNms',category_names);
catIdsToIndex = zeros(max(catIds),1);
catIdsToIndex(catIds) = 1:length(catIds);

save(dst_cat_data_file, 'category_names','catIds','catIdsToIndex');

clear coco;
end

function prepare_test_set_img_data_in_matlab_file(images_directory, img_info_file, dst_images_file)
assert(exist(img_info_file,'file')>0)
img_info =gason(fileread(img_info_file));

img_objs    = img_info.images;
image_names = {img_objs.file_name};
image_sizes = [img_objs.height; img_objs.width]';
image_names = image_names(:);
image_paths = strcat(images_directory, image_names);
num_imgs    = length(image_paths);
imgIds      = [img_objs.id]';
for i = 1:num_imgs
    assert(exist(image_paths{i},'file')>0);
end
save(dst_images_file, 'image_names', 'image_sizes', 'imgIds', '-v7.3');
end

