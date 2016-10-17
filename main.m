%% MLVC - Group 12
% Elisabeth Wetzer
% Lena Trautmann
% Hanna Huber

clear all
close all
clc

%% Task 1.1.
% read in data -> new .csv file written by .py-file
data = dlmread('perceptrondataUseful.csv',','); 

% plot the input vectors in R^2
h1 = scatterData(data,'x','y','input vectors');
% save the figure
printPDF(h1, 'inputVectors');

% do feature transformation
transformed_data = [data(:,1:2).^2 data(:,3)];
% plot again
h2 = scatterData(transformed_data,'x','y','transformed input vectors');
% save the figure
printPDF(h2, 'transformedInputVectors');


%% Task 1.1.2
