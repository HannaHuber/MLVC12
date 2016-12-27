function data = generateTrainingData(N, xRange, yRange, linear)
    
    % generate N random 2D samples 
    X = rand(N, 2);
    
    % map from (0,1) to specified range
    X(:,1) = repmat(xRange(1), N, 1) + X(:,1)*(xRange(2) - xRange(1));
    X(:,2) = repmat(yRange(1), N, 1) + X(:,2)*(yRange(2) - yRange(1));

    % create target labels for linearly separable data
    if linear
        labels = int16(sum(X,2) > mean(xRange) + mean(yRange));
        labels(labels<1) = -1;
        
    % create target labels for general data
    else
        % TODO
        
    end
    
    % assign labels to data points
    data = [ X labels];
    
end