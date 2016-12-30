%% MLVC Assignment 2
% Group 12
close all
%% Init variables
N = 100;                                                                    % number of training samples
xRange = [0 100];                                                           % range of x coords
yRange = [0 100];                                                           % range of y coords

%% 1.1: The dual optimization problem

% create linearly separable data 
[X, t] = generateTrainingData(N, xRange, yRange, true);

% plot data with class labels
h = scatterData([X', t], 'x', 'y', 'Linearly Separable Data');
%printPDF(h, '../figures/linearData.png');

% train support vector machine
alpha = trainSVM(X, t, false);

% find support vectors
idxSV = find(alpha>1e-8);
h = scatterData([X', t], 'x', 'y', 'Support Vectors');
hold on
plot(X(1,idxSV), X(2,idxSV), 'bo');
%printPDF(h, '../figures/sv.png');

%% 1.2 the kernel trick
% write the rbfkernel function and use differen values for sigma




