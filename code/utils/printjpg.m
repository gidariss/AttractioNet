function printjpg(fname)
%PRINTPDF Prints the current figure into a jpg document

set(gca, 'LooseInset', get(gca, 'TightInset'));
print('-djpeg', '-r60', fname);
end
