function [ w, runs ] = lmsTrain( X, t , gamma, online)
%LMSTRAIN Performs online least mean squares learning
%   Input:
%       X       ...     matrix with input vectors in its columns
%       t       ...     vector with target values
%       gamma   ...     learning rate
%       online  ...     true/false for online/batch optimization                
%   Output:
%       w       ...     augmented weight vector (w(1) = bias) corresponding 
%                       to Sum of Squared Error of weighted values and their target
%       runs    ...     number of iterations after which the algorithm
%                       terminates

% get data dimensions
[D, N] = size(X);

% initialize weight vector
w = zeros(1,D);

% execution nr
runs = 0;
maxIter = 10000;

% error
cumErr = 0;
prevAvgErr = 10000;
errRatio = 0;
epsilon = 0.00001;

% Learn until average error over a number of epochs no longer changes
while ((runs < maxIter) && errRatio < 1-epsilon)

    % update w online
    if online
        for i=1:N         
            w = w + gamma*((t(i)-w*X(:,i))*X(:,i))';            
        
        end %for 
        
    % update w in batch form    
    else 
        w = w + gamma*(X*(t - w*X)')';  
    end %if
        
    % update average error and iteration counter
    runs = runs + 1;
    cumErr = cumErr + 0.5*(t - w*X)*(t - w*X)';
    avgErr = cumErr/runs;
    
    % calculate how much the average error has changed and reset variables
    errRatio = (avgErr+1)/(prevAvgErr+1);
    prevAvgErr = avgErr;

end

end

