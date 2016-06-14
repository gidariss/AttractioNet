function evaluateProposalWithCOCO_API(dataset, dataType, imgIds, resFile)
dataType = regexprep(dataType,'_','');
% select results type for demo (either bbox or segm)
% type = {'segm','bbox'}; 
type = 'bbox'; % specify type here
fprintf('Running demo for *%s* results.\n\n',type);

% initialize COCO ground truth api
switch dataset
    case 'mscoco'
        dataDir='./datasets/MSCOCO';
        annFile=sprintf('%s/annotations/instances_%s.json',dataDir,dataType);
    case 'pascal'
        dataDir='./sup_data';
        annFile=sprintf('%s/pascal_%s.json',dataDir,dataType);        
end
if(~exist('cocoGt','var')), cocoGt=CocoApi(annFile); end
cocoDt=cocoGt.loadRes(resFile);

% run Coco evaluation code (see CocoEval.m)
cocoEval=CocoEval(cocoGt,cocoDt);
cocoEval.params.imgIds=imgIds;
cocoEval.params.useSegm=strcmp(type,'segm');
cocoEval.params.useCats = 0;
cocoEval.params.maxDets = [1,10,100,1000];

cocoEval.evaluate();
cocoEval.accumulate();
cocoEval.summarize();
end