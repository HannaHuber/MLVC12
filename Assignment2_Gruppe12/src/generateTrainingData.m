function [X, t] = generateTrainingData(N, xRange, yRange, linear)


% create target labels for linearly separable data
if linear
    % generate N random 2D samples
    X = rand(2, N);
    
    % map from (0,1) to specified range
    X(1, :) = repmat(xRange(1), 1, N) + X(1, :)*(xRange(2) - xRange(1));
    X(2, :) = repmat(yRange(1), 1, N) + X(2, :)*(yRange(2) - yRange(1));
    
    t = double(sum(X) > mean(xRange) + mean(yRange))';
    t(t<1) = -1;
    
% create target labels for general data
else
    % read in data -> new .csv file written by .py-file
    data = dlmread('perceptrondataUseful.csv',',');
    X(1,:) = data(:,1);
    X(2,:) = data(:,2);
    t=data(:,3);
    
end

end