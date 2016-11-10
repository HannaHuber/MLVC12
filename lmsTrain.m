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
<<<<<<< HEAD
gamma = 0.00001;
=======
gamma = 0.000001;
>>>>>>> 9ae1a65c3fe9d172ac8666ce659012e1b976575e

% epsilon_LMS
epsilon = 0.01;

% execution nr
runs = 0;
maxIter = 10000;

% error
prevErr = inf;
currErr = 0.5*(t - w*X)*(t - w*X)';
tau = 1 - 0.0000001;

% just for observation of w:
w_observe = zeros(1,D);

% online LMS:
<<<<<<< HEAD
while (currErr/prevErr < tau) && (runs < maxIter) %correctly classified if all entries of w*X.*t are 1
    for i=1:N
        %if 0.5*(t(i) - w*X(:,i))^2 > epsilon % it is missclassified
            % update w            
            w = w + gamma*((t(i)-w*X(:,i))*X(:,i))';            
        %end
    end  
    w_observe(size(w_observe,1)+1,:) = w; %just for observation reasons
    
    % update for next iteration
=======
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
>>>>>>> 9ae1a65c3fe9d172ac8666ce659012e1b976575e
    runs = runs+1;
    prevErr = currErr;
    currErr = 0.5*(t - w*X)*(t - w*X)';
end

runs
currErr


end

