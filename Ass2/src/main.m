%% MLVC Assignment 2
% Group 12
clear
close all
%% Init variables
N = 100;                                                                    % number of training samples
xRange = [-0.5 0.5];                                                           % range of x coords
yRange = [-0.5 0.5];                                                           % range of y coords

%% 1.1: The dual optimization problem

% create linearly separable data 
[X, t] = generateTrainingData(N, xRange, yRange, true);

% separate in training set (X,t) and test set (X_new,t_new)
X_new = X(:,1:5);
t_new = t(1:5);
X = X(:,6:end);
t = t(6:end);

% plot data with class labels
h = scatterData([X', t], 'x', 'y', 'Linearly Separable Data', 'filled');
% printPDF(h, '../figures/linearData');

% train support vector machine wihtour kernel and slack vars
alpha = trainSVM(X, t, false, 0, 0);

% find support vectors
idxSV = find(alpha>1e-8);
h = scatterData([X', t], 'x', 'y', 'Support Vectors', 'filled');
hold on
plot(X(1,idxSV), X(2,idxSV), 'bo');
getDecisionBoundary(h, alpha, X, t, false, 0);
% printPDF(h, '../figures/sv');

%% 1.2 the kernel trick
% write the rbfkernel function and use different values for sigma

clear X
clear t
% generate non-lin. separable data
[X, t] = generateTrainingData(N, xRange, yRange, false);


% sigma = 8;
sigma = [0.05,0.5,1,4,7.06,7,07];
for i=1:length(sigma)
   
    alpha = trainSVM(X, t, true, sigma(i),0);
       
    % find support vectors
    idxSV = find(alpha>1e-8); 
    h = scatterData([X', t], 'x', 'y', ['Support Vectors; sigma=',num2str(sigma(i))], 'filled');
    hold on
    plot(X(1,idxSV), X(2,idxSV), 'bo');
    getDecisionBoundary(h, alpha, X, t, true, sigma(i));
    printPDF(h, ['../figures/sv_kernel',num2str(sigma(i)*100)]);
end

% classify X_new
y_new_kernel = predictSVM(alpha,X,t,X_new,true,35);
