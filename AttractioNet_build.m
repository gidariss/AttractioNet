function AttractioNet_build()
if ~exist('nms_mex', 'file')
    try
        fprintf('Compiling nms_mex\n');

        mex -outdir bin ...
          -largeArrayDims ...
          code/NMS/nms_mex.cpp ...
          -output nms_mex;
    catch exception
        fprintf('Error message %s\n', getReport(exception));
    end
end
if ~exist('nms_gpu_mex', 'file')
    try
       fprintf('Compiling nms_gpu_mex\n');
       addpath(fullfile(pwd, 'code', 'NMS'));
       nvmex('code/NMS/nms_gpu_mex.cu', 'bin');
       delete('nms_gpu_mex.o');
    catch exception
        fprintf('Error message %s\n', getReport(exception));
    end
end
end
