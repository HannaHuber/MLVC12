function [ w ] = lmsTrain( X, t )
%LMSTRAIN Performs online least mean squares learning

% get data dimensions
[D, N] = size(X);

% initialize weight vector
w = zeros(1,D);

% initialize learning rate
gamma = 0.0001;

% epsilon_LMS
epsilon = 0.01;

% execution nr
runs = 0;

% just for observation of w:
w_observe = zeros(1,D);

% online LMS:
while (0.5*(t - w*X)*(t - w*X)'>epsilon) && (runs < 10000) %correctly classified if all entries of w*X.*t are 1
    for i=1:N
        if 0.5*(t(i) - w*X(:,i))^2 > epsilon % it is missclassified
            % update w            
            w = w + gamma*((t(i)-w*X(:,i))*X(:,i))';            
        end
    end  
    w_observe(size(w_observe,1)+1,:) = w; %just for observation reasons
    runs = runs+1;
end


end

