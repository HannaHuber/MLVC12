function [ b ] = getDecisionBoundary( alpha,X,t,kernel, sigma )
%GETDECISIONBOUNDARY returns decision boundary calculated by a SVM 
% between two classes in a
% dataset X

% create grid in data range
[xgrid, ygrid] = meshgrid(min(X(1,:)):0.1:max(X(1,:)),...
                          min(X(2,:)):0.1:max(X(2,:)) ); 
grid = [xgrid(:)' ; ygrid(:)'];

% predict labels on grid
labels = predictSVM(alpha, X, t, grid, kernel, sigma);

% find boundary
tol = 0.001;
b = labels(abs(labels) < tol);


end

