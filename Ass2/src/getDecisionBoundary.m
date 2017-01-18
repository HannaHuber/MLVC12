function [  ] = getDecisionBoundary( alpha,X,t,kernel, sigma )
%GETDECISIONBOUNDARY returns decision boundary calculated by a SVM 
% between two classes in a dataset X with labels t

% get support vectors
idxSV = find(alpha>1e-8);

% create grid in support vector range
limits = [min(X(:,idxSV), [], 2) max(X(:,idxSV), [], 2)];
step = diff(limits,1,2)/100.0;
[xgrid, ygrid] = meshgrid(limits(1,1):step(1):limits(1,2),...
                          limits(2,1):step(2):limits(2,2) ); 
grid = [xgrid(:)' ; ygrid(:)'];

% predict labels on grid
labels = predictSVM(alpha, X, t, grid, kernel, sigma)';

% plot prediction boundary
scatterData([xgrid(:), ygrid(:), sign(labels)], 'x', 'y', 'Prediction', '+');
hold on
contour(xgrid, ygrid,reshape(labels, size(xgrid)),[0,0])

end

