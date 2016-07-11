function conf = AttractioNet_get_defaul_conf(varargin)
% AttractioNet_get_defaul_conf returns the default configuration options of
% the AttractioNet model.
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
ip.addParamValue('nms_iou_thrs',   [0.95, 0.90, 0.85, 0.80, 0.75, 0.65, 0.60, 0.55],  @isnumeric);
ip.addParamValue('max_per_image',  [2000, 1000,  400,  200,  100,   40,   20,   10],  @isnumeric);
ip.addParamValue('num_iterations',         5,  @isscalar);  % number of iterations.
ip.addParamValue('num_seed_boxes',     10000,  @isscalar);  % number of seed boxes.
ip.addParamValue('max_per_image_init',  2000,  @isscalar);  % number of boxes that will be kept after the first iteration.
ip.addParamValue('iou_thrs_close',      0.90,  @isscalar);  
% iou_thrs_close: is an iou threshold that is being used in the fast version 
% of the algorithm in order to suppress candide box proposals that are very 
% close to each other (their IoU is greater or equal to iou_thrs_close) 
% in order to avoid performing unnecessary computations.

% number of output box proposals
ip.addParamValue('num_output_boxes',    2000,  @isscalar);  
ip.addParamValue('scales',             1000,   @isnumeric);  % the shortest dimension of the image before is fed to the proposal model
ip.addParamValue('max_size',           1400,   @isscalar); % maximum size of the lon
% during test time, the image is scaled such that its shortest dimension to be 
% 'scales' pixels taking care however its longest dimension to not exceed
% the 'max_size' pixels.

ip.parse(varargin{:});
conf = ip.Results;
end

