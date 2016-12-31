function [ K ] = rbfkernelMatrix( X1,X2,sigma )
%RBFKERNELMATRIX calculates the radial basis function kernel of the two input
%vector matrices and the RBF parameter:
%   K(x,y) = exp(- ||x-y||^2 / sigma^2)
%
%   Input:
%       X1  ...         1st input matrix [dxN] d... dimension, N... no of
%       data points
%       X2  ...         2nd input vector [dxN]
%       sigma   ...     RBF parameter [scalar]
%   Output:
%       K   ...         kernel [NxN]
%

K = zeros(size(X1,2), size(X2,2));
for i1 = 1:size(X1,2)
    for i2 = 1:size(X2,2)
        K(i1,i2) = rbfkernel(X1(:,i1),X2(:,i2),sigma);
    end
end
end

