function [ h ] = scatterData( data , xlab, ylab, tit )
%SCATTERDATA special scatter function for our type of data
%   Input:
%       data:   1st column is x, 2nd y, 3rd is target value for the color
%       xlab: xlabel in the figure
%       ylab: ylabel in the figure
%       tit:  title in the figure
%   
%   Output:
%       h:      figure handle

h = figure()
scatter(data(:,1),data(:,2),20,data(:,3),'filled')
xlabel(xlab)
ylabel(ylab)
title(tit)
end

