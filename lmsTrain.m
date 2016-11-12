function [ w ] = lmsTrain( X, t , online)
%LMSTRAIN Performs online least mean squares learning
%   Input:
%       X       ...     matrix with input vectors in its columns
%       t       ...     vector with target values
%       online  ...     true/false for online/batch optimization                
%   Output:
%       w       ...     augmented weight vector (w(1) = bias) corresponding 
%                       to Sum of Squared Error of weighted values and their target

% get data dimensions
[D, N] = size(X);

% initialize weight vector
w = zeros(1,D);

% initialize learning rate
gamma = 0.0001;

% execution nr
runs = 0;
maxIter = 10000;

% error
currErr = 0;
prevAvgErr = inf;
errRatio = 0;
tau = 1 - 0.00001;

% just for observation of w:
w_observe = zeros(1,D);

% Learn until average error over a number of epochs no longer changes
while ((errRatio < tau) && (runs < maxIter))
    
    % update w online
    for i=1:N         
            w = w + gamma*((t(i)-w*X(:,i))*X(:,i))';            
        %end
    end  

    % just for observation reasons
    w_observe(size(w_observe,1)+1,:) = w; 

    % update average error and iteration counter
    runs = runs + 1;
    currErr = currErr + 0.5*(t - w*X)*(t - w*X)'
    avgErr = currErr/runs
    
    % calculate how much the average error has changed and reset variables
    errRatio = avgErr/prevAvgErr;
    prevAvgErr = avgErr;

end

end

