function [ alpha ] = trainSVM( X, t , kernel, sigma)
%TRAINSVM trains a support vector machine (svm)
%   Input:
%       X   ...     2xN matrix of N 2D training samples
%       t   ...     Nx1 vector of corresponding class labels
%       kernel ...  boolean: 0 = no kernel used; 1 = RBF-kernel used.
%       sigma ...   rbf-parameter
%   Output:
%       alpha   ...     Nx1 vector of lagrangian multipliers defining svm
%                       alpha(i) > 0 if X(:, i) is a support vector

% eqs. 42+43: maximize 
%       -1/2 * sum[alpha(i)alpha(j)t(i)t(j)(x(i)'x(j))] + % sum[alpha(i)],
%       where
%       sum[alpha(i)t(i)] = 0
%       alpha(i) >=0

% quadprog: minimize 
%       1/2*x'*H*x + f'*x subject to the restrictions 
%       A*x <= b, 
%       Aeq*x = beq, 
%       lb <= x <= ub

% define constraint parameters  (negative for minimization)
if kernel % RBF-kernel
%     sigma = 10;
    H = diag(t) * rbfkernelMatrix(X,X,sigma) * diag(t);
else % no kernel
    H = diag(t) * (X') * X * diag(t);
end
f = (-1) * ones(size(t,1),1);
A = [];
b = [];
Aeq = t';
beq = 0;
lb = zeros(size(t,1),1);
ub = [];
x0 = [];
options = optimoptions(@fmincon,'Algorithm', 'interior-point');

% calculate lagrange multipliers
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options);


end

