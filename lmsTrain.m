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
gamma = 0.000001;

% epsilon_LMS
epsilon = 0.01;

% execution nr
runs = 0;

% just for observation of w:
w_observe = zeros(1,D);

% online LMS:
while (0.5*(t - w*X)*(t - w*X)'>epsilon) && (runs < 10000)
    if online
        for i=1:N
            if 0.5*(t(i) - w*X(:,i))^2 > epsilon % it is missclassified
                % update w            
                w = w + gamma*((t(i)-w*X(:,i))*X(:,i))';            
            end
        end  
        w_observe(size(w_observe,1)+1,:) = w; %just for observation reasons
    else %batch
        % find misclassified input vectors
        misclas = (0.5*(t - w*X).*(t - w*X)>epsilon);
        % update their weight
        w = w + gamma*((t(misclas)-w*X(:,misclas))*X(:,misclas)');
    end
    runs = runs+1;
end


end

