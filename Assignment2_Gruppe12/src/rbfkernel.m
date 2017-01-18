function [ k ] = rbfkernel( x1,x2,sigma )
%RBFKERNEL calculates the radial basis function kernel of the two input
%vectors and the RBF parameter:
%   K(x,y) = exp(- ||x-y||^2 / sigma^2)
%
%   Input:
%       x1  ...         1st input vector [dx1]
%       x2  ...         2nd input vector [dx1]
%       sigma   ...     RBF parameter [scalar]
%   Output:
%       k   ...         kernel [scalar]
%

k = exp(-norm(x1-x2,2)^2/sigma^2); 
end

