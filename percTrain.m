function [ w ] = percTrain( X, t, maxIts, online )
%PERCTRAIN Calculates perceptron decision boundary.
%   Input:
%       X       ...     matrix with input vectors in its columns
%       t       ...     vector with target values
%       maxIts  ...     upper limit for iterations of the gradient-based
%                       optimization procedure
%       online  ...     true/false for online/batch optimization                
%   Output:
%       w       ...     augmented weight vector (w(1) = bias) corresponding 
%                       to decision boundary separating the input vectors 
%                       according to their target values

    % homogeneous coords (X(1, :) = 1)
    [d, n] = size(X);
    X = [ones(1, n); X ];
    
    % init 
    w = zeros(d+1, 1);  % weight vector 
    gamma = 1;          % learning rate
    itCount = 0;        % iteration counter
    
    % online learning
    if (online)
        
        % train until all vectors are correctly classified or the maximim
        % number of iterations is reached
        while (any((w' * X) .* t' <= 0) && itCount < maxIts)

            % consider input vectors consecutively
            for i = 1:n

                % update weight vector in case of misclassified sample
                if (w' * X(:, i) * t(i) <= 0)

                    % add sample scaled by learning rate
                    w = w + gamma * X(:, i) * t(i);

                end

            end

            % update iteration counter
            itCount = itCount + 1;
            
            % plot results at different stages

        end
    
    % batch learning
    else
        
        % multiplay input vectors with labels for later updates
        samples = X .* repmat(t', d+1, 1);
        
        % train until all vectors are correctly classified or the maximim
        % number of iterations is reached
        while (any((w' * X) .* t' <= 0) && itCount < maxIts)

            % collect misclassified samples
            classified = w' * samples;
            misclassified = classified <=0;
            deltaW = sum(samples(:, misclassified), 2);
            
            % update weight vector according to learning rate
            w = w + gamma * deltaW;
                    
            % update iteration counter
            itCount = itCount + 1;
            
        end
    end
    

    
end

