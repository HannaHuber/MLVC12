function [ h ] = scatterData( data , xlab, ylab, tit, mkr )
%SCATTERDATA special scatter function for our type of data
%   Input:
%       data:   1st column is x, 2nd y, 3rd is target value for the color
%       xlab: xlabel in the figure
%       ylab: ylabel in the figure
%       tit:  title in the figure
%   
%   Output:
%       h:      figure handle

h = figure();
cols = zeros(size(data, 1), 3);
szClass1 = nnz(data(:,3)>0);
cols(data(:,3)>0, :) = repmat([1 0 0], szClass1, 1);
cols(data(:,3)<0, :) = repmat([0 1 0], size(data, 1) - szClass1, 1);
scatter(data(:,1),data(:,2),20,cols,mkr)
xlabel(xlab)
ylabel(ylab)
title(tit)
end

