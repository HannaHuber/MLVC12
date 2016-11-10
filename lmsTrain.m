function [ w ] = lmsTrain( X, t )
%LMSTRAIN Performs online least mean squares learning

% get data dimensions
[D, N] = size(X);

% initialize weight vector
w = zeros(1,D);

% initialize learning rate
gamma = 0.00001;

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
while (currErr/prevErr < tau) && (runs < maxIter) %correctly classified if all entries of w*X.*t are 1
    for i=1:N
        %if 0.5*(t(i) - w*X(:,i))^2 > epsilon % it is missclassified
            % update w            
            w = w + gamma*((t(i)-w*X(:,i))*X(:,i))';            
        %end
    end  
    w_observe(size(w_observe,1)+1,:) = w; %just for observation reasons
    
    % update for next iteration
    runs = runs+1;
    prevErr = currErr;
    currErr = 0.5*(t - w*X)*(t - w*X)';
end

runs
currErr


end

