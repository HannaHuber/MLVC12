function [ w, runs ] = lmsTrainGamma( X, t , gamma, online)
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
wDiff = inf;
wstar = (pinv(X')*t')';
epsilon = 0.5;

% Learn until average error over a number of epochs no longer changes
while ((runs < maxIter) && wDiff > epsilon)

    % update w online
    if online
        for i=1:N         
            w = w + gamma*((t(i)-w*X(:,i))*X(:,i))';            
        
        end %for 
        
    % update w in batch form    
    else 
        w = w + gamma*(X*(t - w*X)')';  
    end %if
    
    % eukledian norm of the distance between the current and the exact
    % weight vector
    wDiff = norm(w-wstar);
    
    % update iteration counter
    runs = runs + 1;

end

end

