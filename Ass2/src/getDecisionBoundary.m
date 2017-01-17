function [ b ] = getDecisionBoundary( alpha,X,t,kernel, sigma )
%GETDECISIONBOUNDARY returns decision boundary calculated by a SVM 
% between two classes in a dataset X with labels t

% get convex hull of svs
idxSV = find(alpha>1e-8);
sv = X(:,idxSV);
hull = convhull(sv');

% create grid in within hull of svs
[xgrid, ygrid] = meshgrid(min(X(1,idxSV)):0.1:max(X(1,idxSV)),...
                          min(X(2,idxSV)):0.1:max(X(2,idxSV)) ); 
in = inpolygon(xgrid, ygrid, sv(1, hull), sv(2, hull));
grid = [xgrid(in)' ; ygrid(in)'];

% predict labels on grid
labels = predictSVM(alpha, X, t, grid, kernel, sigma);

% find boundary
tol = 0.001;
b = labels(abs(labels) < tol);

end

