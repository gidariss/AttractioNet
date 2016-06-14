function [all_bboxes, dst_box_proposal_paths] = AttractioNet_run_on_dataset(model, image_paths, dst_dir, varargin)
% AttractioNet_run_on_dataset extracts the AttractioNet box proposals from
% a set of images and saves them (using a single matlab file per image) 
% under the specified destination directory.
% 
% INPUTS:
% (1) model: the model in the form of a matlab struct data type
% (2) image_paths: a N x 1 cell array of strings with the image file paths.
% N is the number of images.
% (3) dst_dir: string with the destination directory under which the
% produced matlab files with the extracted AttractioNet box proposals will 
% be stored. Specifically, the produced files will be saved on the 
% location: [dst_dir,'/boxes/']
% (4) conf: struct with the AttractioNet configuration options.
% 
% OUTPUTS:
% (1) all_bboxes: is a N x 1 cell array with the AttractioNet box proposals of 
% each input image. Specifically, all_bboxes{i} is a K x 5 single precision 
% numeric array with the box proposals of the i-th image image_paths{i} 
% (where K is the number of box proposals; by default 2000).
% (2) all_bboxes_filepaths: is a N x 1 cell array with the paths to the matlab
% files that contain the AttractioNet extracted box proposals. Specifically,
% all_bboxes_filepaths{i} is a string of the path to the matlab file that
% contains the box proposals of the i-th image image_paths{i}. Note that
% the matlab files have the same filenames as the original images.
% 
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

ip = inputParser;
ip.addParamValue('conf', struct);

ip.parse(varargin{:});
opts = ip.Results;

dst_dir_boxes = fullfile(dst_dir,'boxes/');
dst_dir_log   = fullfile(dst_dir,'log');
mkdir_if_missing(dst_dir);
mkdir_if_missing(dst_dir_boxes);
mkdir_if_missing(dst_dir_log);

timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_file  = fullfile(dst_dir_log, ['log_file_', timestamp, '.txt']);

dst_box_proposal_paths = strcat(dst_dir_boxes, getImageIdsFromImagePaths( image_paths ),'.mat');

t_start = tic();
diary(log_file);
all_bboxes = process_images(opts.conf, model, image_paths, dst_box_proposal_paths);
diary off;
fprintf('Extract bounding box proposals in %.4f minutes.\n', toc(t_start)/60);
end


function [all_bboxes] = process_images(conf, model, image_paths, dst_box_proposal_paths)
checkpoint_step = 500;
model.max_rois_num_in_gpu = 1000;%find_max_rois_num_in_gpu(model); 
num_imgs   = length(image_paths);
all_bboxes = cell(length(image_paths),1);
total_el_time = 0; total_pr_time = 0; counter = 0; num_chars = 0;

for img_idx = 1:num_imgs
    
    try 
        % try to load the proposals from the disk
        all_bboxes{img_idx} = load_bbox_proposals(dst_box_proposal_paths{img_idx});
        processing_time = 0; elapsed_time = 0;
    catch
        th_all = tic;
        %************** EXTRACT BOX PROPOSALS FROM THIS IMAGE *************    
        image = get_image(image_paths{img_idx}); 
        th = tic;
        all_bboxes{img_idx} = AttractioNet(model, image, conf);
        processing_time = toc(th); 
        %******************************************************************
        save_bbox_proposals(dst_box_proposal_paths{img_idx},all_bboxes{img_idx}); % save box proposals
        counter = counter + 1;
        elapsed_time = toc(th_all);   
    end
    
    if (mod(img_idx, checkpoint_step) == 0)
        diary; diary; % flush diary
    end
    
    %*************************** PRINT PROGRESS ***************************
    [total_pr_time, avg_pr_time] = timing_process(processing_time, total_pr_time, 1, counter, num_imgs);    
    [total_el_time, avg_time, est_rem_time] = timing_process(elapsed_time, total_el_time, 1, counter, num_imgs);
    fprintf(repmat('\b',[1, num_chars]));
    num_chars = fprintf('%s: extract box proposals %d/%d:| ET %.3fs PT %.3f | AET: %.3fs APT %.3fs | TT %.4fmin | ERT %.4fmin |\n', ...
        procid(), img_idx, num_imgs, elapsed_time, processing_time, avg_time, avg_pr_time, total_el_time/60, est_rem_time/60);  
    %**********************************************************************    
end

end

function [total_el_time, ave_time, est_rem_time] = timing_process(...
    elapsed_time, total_el_time, fist_img_idx, i, num_imgs)
total_el_time   = total_el_time + elapsed_time;
ave_time        = total_el_time / (i-fist_img_idx+1+eps);
est_rem_time    = ave_time * (num_imgs - i);
end