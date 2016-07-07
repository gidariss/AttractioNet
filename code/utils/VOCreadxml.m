% This code was originally written and distributed as part of the
% PASCAL VOC challenge
function rec = VOCreadxml(path)

if length(path)>5&&strcmp(path(1:5),'http:')
    xml=urlread(path)';
else
    f=fopen(path,'r');
    xml=fread(f,'*char')';
    % added by hyli
    if strcmp(xml(1:2), '<?')
        xml = xml(23:end);
    end
    fclose(f);
end
rec=VOCxml2struct(xml);
