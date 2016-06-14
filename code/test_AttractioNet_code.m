function test_AttractioNet_code( gpu_id )
%************************* SET GPU/CPU DEVICE *****************************
% By setting gpu_id = 1, the first GPU (one-based counting) is used for 
% running the AttractioNet model. 
if ~exist('gpu_id','var'), gpu_id = 1; end
caffe_set_device( gpu_id );
caffe.reset_all();
%**************************************************************************
%***************************** LOAD MODEL *********************************
model_dir_name = 'AttractioNet_Model';
full_model_dir = fullfile(pwd, 'models-exps', model_dir_name); 
assert(exist(full_model_dir,'dir')>0,sprintf('The %s model directory does not exist',full_model_dir));
mat_file_name  = 'box_proposal_model.mat';
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
%************************ TEST AttractioNet CODE **************************
tolerance  = 10^(-4);
image_path{1} = fullfile(pwd,'examples','COCO_val2014_000000109798.jpg');
fprintf('--------Testing the current state of AttractioNet code------------\n');
for i = 1:length(image_path)
    fprintf('Check image %d / %d',i,length(image_path))
    image_path_this = image_path{i};
    % run AttractioNet code in its current state in order to create box 
    % proposals
    image = imread(image_path_this);
    tic;
    bbox_proposals = AttractioNet(model, image, box_prop_conf);
    elapsed_time = toc;
    fprintf(' elapsed time %.3f secs:', elapsed_time);
    % read the "correct" box proposals generated from a known to work
    % version of the AttractioNet code
    [file_dir,filename] = fileparts(image_path_this);
    proposals_path = fullfile(file_dir,[filename,'.mat']);
    bbox_proposals_correct = load_bbox_proposals(proposals_path);
    
    overlaps = boxoverlap(bbox_proposals, bbox_proposals_correct);
    abo_all  = 100 * mean(max(overlaps,[],2));
    abo_1000 = 100 * mean(max(overlaps(1:1000,1:1000),[],2));
    abo_100  = 100 * mean(max(overlaps(1:100,1:100),  [],2));
    abo_10   = 100 * mean(max(overlaps(1:10,1:10),    [],2));
   
    min_abo = min([abo_all,abo_1000,abo_100,abo_10]);
    % Check if the proposals produced from the current code match the 
    % "correct" box proposals that AttractioNet should return.
    %abs_diffs = abs(bbox_proposals(:) - bbox_proposals_correct(:));
    %num_non_valide_entries = sum(abs_diffs > tolerance);
    if min_abo < 98.0 
        fprintf(' FAIL :-(. Report:\n');
        %fprintf('\tElement wise comparison of the produced boxes with the "correct" boxes:\n')
        %fprintf('\t\t%d (%.4f percent) entries are different more than %e\n',...
        %    num_non_valide_entries,100*num_non_valide_entries/length(abs_diffs), tolerance)
        fprintf('\tAverage best overlap comparison of the top {all,1000,100,10} produced and "correct" boxes:\n')
        fprintf('\t\tAll: %.2f - Top 1000: %.2f - Top 100: %.2f - Top 10: %.2f\n',...
            abo_all, abo_1000, abo_100, abo_10);
        fprintf('\tNote that "correct" boxes are boxes that have been produced from a known to work version of AttractioNet code.\n')
    else
        fprintf(' PASS :-)\n');
    end
end
%**************************************************************************
%********************* FREE GPU/CPU MEMORY (Caffe) ************************
caffe.reset_all();
%**************************************************************************
end

