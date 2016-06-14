function startup()
curdir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(curdir,  'code')));
mkdir_if_missing(fullfile(curdir, 'bin'));
mkdir_if_missing(fullfile(curdir, 'models-exps'));
mkdir_if_missing(fullfile(curdir, 'box_proposals'));

addpath(fullfile(curdir, 'bin'));

caffe_path = fullfile(curdir, 'external', 'caffe_AttractioNet', 'matlab');
if exist(caffe_path, 'dir') == 0
    error('matcaffe is missing from external/caffe_AttractioNet/matlab; See README.md');
end
addpath(genpath(caffe_path));

end