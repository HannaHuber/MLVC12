%% MLVC Assignment 2
% Group 12
%% Init variables
N = 100;                                                                    % number of training samples
xRange = [0 100];                                                           % range of x coords
yRange = [0 100];                                                           % range of y coords

%% 1.1: Create linearly separable data 
data = generateTrainingData(N, xRange, yRange, true);
h = scatterData(data, 'x', 'y', 'Linearly Separable Data');
printPDF(h, '../figures/linearData.png');