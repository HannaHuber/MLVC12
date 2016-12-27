function [ alpha ] = trainSVM( X, t )
%TRAINSVM trains a support vector machine (svm)
%   Input:
%   X   ...     2xN matrix of N 2D training samples
%   t   ...     Nx1 vector of corresponding class labels
%   Output:
%   alpha   ...     Nx1 vector of lagrangian multipliers defining svm
%                   alpha(i) > 0 if X(:, i) is a support vector

% eqs. 42+43: maximize 
%       -1/2 * sum[alpha(i)alpha(j)t(i)t(j)(x(i)'x(j))] + % sum[alpha(i)],
%       where
%       sum[alpha(i)t(i)] = 0
%       alpha(i) >=0

% quadprog: minimize 
%       1/2*x'*H*x + f'*x subject to the restrictions 
%       A*x ? b, 
%       Aeq*x = beq, 
%       lb ? x ? ub

% define constraint parameters  (negative for minimization)
H = diag(t) * (X') * X * diag(t);
f = (-1) * ones(size(t,1),1);
A = [];
b = [];
Aeq = t';
beq = 0;
lb = zeros(size(t,1),1);
ub = inf;

% calculate lagrange multipliers
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub);


end

