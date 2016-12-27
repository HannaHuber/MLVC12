function [ y_new ] = predictSVM( alpha, X, t, x_new )
%PREDICTSVM Classifies new data using a trained support vector machine
%
%   

% find support vectors
idxSV = find(alpha>0);

% calculate w0
w0 = t(idxSV(1)) - alpha' * diag(t) * X' * X(:,idxSV(1));

% classify new data (eq.12)
y_new = alpha' * t * X' * x_new + w0;

end

