function res = prepareCOCOStyleBoxPropResults(image_db, all_bbox_props, json_file)
image_paths = image_db.image_paths;
imgIds      = image_db.imgIds;

assert(length(all_bbox_props) == 1);
assert(length(all_bbox_props{1}) == length(image_paths));
assert(length(all_bbox_props{1}) == length(imgIds));

res = prepareResultsStructFast(all_bbox_props, imgIds);

if (exist('json_file','var')>0) && ~isempty(json_file)
    f=fopen(json_file,'w'); fwrite(f,gason(res)); fclose(f); 
end

end

function results = prepareResultsStructFast(all_bbox_props, imgIds)
num_categories = length(all_bbox_props);
assert(num_categories == 1);
num_images = length(all_bbox_props{1});

field_image_id_all    = {};
field_category_id_all = {};
field_bbox_all        = {};
score_all             = {};

counter = 0;
counter2 = 0;

cat_idx = 1;
category_id = 1;
for img_idx = 1:num_images
    image_id = uint32(imgIds(img_idx));

    num_dets = size(all_bbox_props{cat_idx}{img_idx},1);
    if (num_dets > 0)
        field_image_id    = repmat(image_id,    [num_dets,1]);
        field_category_id = repmat(category_id, [num_dets,1]); 
        field_bbox        = single(all_bbox_props{cat_idx}{img_idx}(:,1:4));
        field_bbox        = [field_bbox(:,1)-1,field_bbox(:,2)-1,field_bbox(:,3)-field_bbox(:,1)+1,field_bbox(:,4)-field_bbox(:,2)+1];
        score             = single(all_bbox_props{cat_idx}{img_idx}(:,5));

        counter                        = counter + 1;
        field_image_id_all{counter}    = field_image_id;
        field_category_id_all{counter} = field_category_id;
        field_bbox_all{counter}        = field_bbox;
        score_all{counter}             = score;
    end
    counter2 =  counter2 + 1;
    tic_toc_print('setup img data %d / % d (%.2f)\n', counter2, (num_images*num_categories), 100*counter2/(num_images*num_categories));
end
    
field_image_id_all    = num2cell(cell2mat(field_image_id_all(:)),2);
field_category_id_all = num2cell(cell2mat(field_category_id_all(:)),2);
field_bbox_all        = num2cell(cell2mat(field_bbox_all(:)),2);
score_all             = num2cell(cell2mat(score_all(:)),2);

results = cell2struct(...
    [field_image_id_all, field_category_id_all, field_bbox_all, score_all], ...
    {'image_id',         'category_id',         'bbox',         'score'}, 2);
end