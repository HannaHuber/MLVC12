function [ y ] = perc( w, X )
%PERC Simulates a perceptron.
%   Input:
%       w   ...     augmented weight vector (w(1) = bias)
%       X   ...     matrix with input vectors in its columns
%   Output:
%       y   ...     binary vector with class labels 1 or -1

    % homogeneous coords (X(1, :) = 1)
    X = [ones(1, size(X, 2)); X ];
    
    % calculate class label (w^t * X >= 0 ... label=1)
    y = sign(w' * X);
    y(y==0) = 1;
    
end

