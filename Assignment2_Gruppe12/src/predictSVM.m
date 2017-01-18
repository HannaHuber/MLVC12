function [ y_new ] = predictSVM( alpha, X, t, x_new, kernel, sigma )
%PREDICTSVM Classifies new data using a trained support vector machine
%   Input:
%       alpha   ... Nx1 vector of lagrangian multipliers defining svm
%                       alpha(i) > 0 if X(:, i) is a support vector 
%       X   ...     2xN matrix of N 2D training samples
%       t   ...     Nx1 vector of corresponding class labels
%       x_new   ... 2xM matrix of data to be classified
%       kernel ...  boolean: 0 = no kernel used; 1 = RBF-kernel used.
%       sigma ...   rbf-parameter
%   Output:
%       y_new ...   1xM vector containing classification of x_new
%       

% find support vectors
idxSV = find(alpha>1e-8);

if kernel
    % calculate w0
    w0 = t(idxSV(1)) - alpha' * diag(t) * rbfkernelMatrix(X,X(:,idxSV(1)),sigma);    
    
    % classify new data (eq.12)
    y_new = (alpha.*t)' *rbfkernelMatrix(X,x_new,sigma) + w0;
    
else %no kernel
    % calculate w0
    w0 = t(idxSV(1)) - alpha' * diag(t) * X' * X(:,idxSV(1));    
    
    % classify new data (eq.12)
    y_new = (alpha.*t)' * X' * x_new + w0;
end
end

