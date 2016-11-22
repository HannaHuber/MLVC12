function [ w, itCount ] = percTrain( X, t, maxIts, online )
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
%       itCount ...     number of iterations after which the algorithm
%                       terminates


    % homogeneous coords (X(1, :) = 1)
    [d, n] = size(X);
    X = [ones(1, n); X ];
    XTransformed = X.^2;
    xyMinTransformed = min(XTransformed(2:3, :),[],2);
    xyMaxTransformed = max(XTransformed(2:3, :),[],2);
    
    
    % init 
    w = zeros(d+1, 1);  % weight vector 
    gamma = 1;          % learning rate
    itCount = 0;        % iteration counter
    
    % online learning
    if (online)
        
        % train until all vectors are correctly classified or the maximim
        % number of iterations is reached
        while (any((w' * XTransformed) .* t' <= 0) && itCount < maxIts)

            % consider input vectors consecutively
            for i = 1:n

                % update weight vector in case of misclassified sample
                if (w' * XTransformed(:, i) * t(i) <= 0)

                    % add sample scaled by learning rate
                    w = w + gamma * XTransformed(:, i) * t(i);

                end

            end

            % update iteration counter
            itCount = itCount + 1;
            
            % plot results at different stages
            if (itCount==1 || itCount == 3 || itCount == 6)
                h=scatterData([XTransformed(2:end, :)' t], 'x^2', 'y^2', strcat('decision boundary after ', num2str(itCount), ' iteration(s) of online learning'));
                hold on
                plotDecisionBoundary(w, xyMinTransformed(1), xyMinTransformed(2), xyMaxTransformed(1), xyMaxTransformed(2), false);
                %printPDF(h, strcat('../figures/transformedOnlineIt', num2str(itCount)));
                
                h=scatterData([(X(2:end, :)') t], 'x', 'y', strcat('decision boundary after ', num2str(itCount), ' iteration(s) of online learning'));
                hold on
                plotDecisionBoundary(w, xyMinTransformed(1), xyMinTransformed(2), xyMaxTransformed(1), xyMaxTransformed(2), true);
                %printPDF(h, strcat('../figures/originalOnlineIt', num2str(itCount)));
            end

        end
    
    % batch learning
    else
               
        % multiplay input vectors with labels for later updates
        samples = XTransformed .* repmat(t', d+1, 1);
        
        % train until all vectors are correctly classified or the maximim
        % number of iterations is reached
        while (any((w' * XTransformed) .* t' <= 0) && itCount < maxIts)

            % collect misclassified samples
            classified = w' * samples;
            misclassified = classified <=0;
            deltaW = sum(samples(:, misclassified), 2);
            
            % update weight vector according to learning rate
            w = w + gamma * deltaW;
                    
            % update iteration counter
            itCount = itCount + 1;
            
            % plot results at different stages
            if (itCount==1 || itCount == 342 || itCount == 685)
                h=scatterData([XTransformed(2:end, :)' t], 'x^2', 'y^2', strcat('decision boundary after ', num2str(itCount), ' iteration(s) of batch learning'));
                hold on
                plotDecisionBoundary(w, xyMinTransformed(1), xyMinTransformed(2), xyMaxTransformed(1), xyMaxTransformed(2), false);
                %printPDF(h, strcat('../figures/transformedBatchIt', num2str(itCount)));
                
                h=scatterData([(X(2:end, :)') t], 'x', 'y', strcat('decision boundary after ', num2str(itCount), ' iteration(s) of batch learning'));
                hold on
                plotDecisionBoundary(w, xyMinTransformed(1), xyMinTransformed(2), xyMaxTransformed(1), xyMaxTransformed(2), true);
                %printPDF(h, strcat('../figures/originalBatchIt', num2str(itCount)));
            end
            
        end
    end
    

    
end

function [] = plotDecisionBoundary(w, xMin, yMin, xMax, yMax, transform)
    if (w(3) ~=0 )
        p1 = [xMin, (-w(2)*xMin - w(1))/w(3)];
        p2 = [xMax, (-w(2)*xMax - w(1))/w(3)];                
    else
        p1 = [-w(1)/w(2) yMin];
        p2 = [-w(1)/w(2), yMax];
    end
        decBdry = repmat(p1', 1, 101) + repmat((0:0.01:1), 2, 1).*repmat(p2'-p1', 1, 101);
    if transform
        decBdryPos = decBdry.^0.5;
        decBdryNeg = -decBdry.^0.5;
        plot(decBdryPos(1,:), decBdryPos(2,:), 'b--')
        plot(decBdryNeg(1,:), decBdryNeg(2,:), 'b--')
        plot(decBdryPos(1,:), decBdryNeg(2,:), 'b--')
        plot(decBdryNeg(1,:), decBdryPos(2,:), 'b--')
    else
        plot(decBdry(1,:), decBdry(2,:), 'b--')
    end
    

end

