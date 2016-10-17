function [  ] = printPDF( h , filename )
%PRINTPDF adapts PDF-page size to figure
%   no huge blank spaces are around the figure as it fills the page and
%   page size is adapted to the figure dimensions
%
%   Input:
%       h:          figure handle
%       filename:   filename of the figure (adapted to our situation -> it
%                   prints in the figures folder

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,['figures/' filename],'-dpdf','-r0')
end

