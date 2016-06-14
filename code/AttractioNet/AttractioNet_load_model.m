function model = AttractioNet_load_model(full_model_dir, mat_file_name)
% AttractioNet_load_model: loads the AttractioNet model
%
% INPUTS:
% full_model_dir: string with the full path to them model directory.
% mat_file_name : string with the .mat filename of the model
%
% OUTPUT:
% model : the model in the form of a matlab struct data type
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



full_model_path = fullfile(full_model_dir, mat_file_name); % full path of the model mat file
assert(exist(full_model_dir,'dir')>0, sprintf('The model directory: %s does not exist', full_model_dir));
assert(exist(full_model_path,'file')>0, sprintf('The model .mat filename: %s does not exist', full_model_path));


ld = load(full_model_path, 'model'); 
model = ld.model; 
clear ld; 

% load model to caffe
curr_dir = pwd;
cd(full_model_dir);
model.net = caffe_load_model(model.net_def_file, model.net_weights_file, 'test');
cd(curr_dir);
end