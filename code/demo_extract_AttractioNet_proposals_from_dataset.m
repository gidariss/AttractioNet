% Examples of AttractioNet_extract_box_proposals_from_dataset() usage.

% It first extracts AttractioNet box proposals (using the 1st GPU) from the
% PASCAL VOC 2007 test set images and then evaluates their average recall
% performance.
AttractioNet_extract_box_proposals_from_dataset('AttractioNet_Model',...
    'gpu_id',1,'dataset','pascal','set_name','test_2007','eval_props',true);


% It first extracts AttractioNet box proposals (using the 1st GPU) from the
% first 5k images of COCO val 2014 set and then evaluates their average 
% recall performance.
AttractioNet_extract_box_proposals_from_dataset('AttractioNet_Model',...
    'gpu_id',1,'first_5k_coco_val_2014', true,'eval_props',true);

% It extracts the AttractioNet box proposals (using the 1st GPU) from the
% COCO val 2014 set images.
AttractioNet_extract_box_proposals_from_dataset('AttractioNet_Model',...
    'gpu_id',1,'dataset','mscoco','set_name','val_2014');

% It extracts AttractioNet box proposals (using the 1st GPU) from the 1st 
% till the 1000-th image of COCO val 2014 set. 
AttractioNet_extract_box_proposals_from_dataset('AttractioNet_Model',...
    'gpu_id',1,'dataset','mscoco','set_name','val_2014',...
    'startIdx',1,'stopIdx',1000);
