%% MLVC - Group 12
% Elisabeth Wetzer
% Lena Trautmann
% Hanna Huber

%clear all
close all
clc

%% Task 1.1.1 Reading data
% read in data -> new .csv file written by .py-file
data = dlmread('perceptrondataUseful.csv',','); 

% plot the input vectors in R^2
%h1 = scatterData(data,'x','y','input vectors');
% save the figure
%printPDF(h1, 'inputVectors');

% do feature transformation
transformed_data = [data(:,1:2).^2 data(:,3)];
% plot again
%h2 = scatterData(transformed_data,'x','y','transformed input vectors');
% save the figure
%printPDF(h2, 'transformedInputVectors');


%% Task 1.1.2 Perceptron training algorithm
% train perceptron (online + batch learning)
%wOnline = percTrain(data(:,1:2)', data(:,3), 1000, true);
%wBatch = percTrain(data(:,1:2)', data(:,3), 1000, false);

%% Task 1.2.1 Experimental setup
% x as row vector
x = (0:0.1:5);
y = 2*x.^2-12*x+1;
trainingIdx = (1:8:51);
N = length(trainingIdx);
noise = normrnd(0,16,size(trainingIdx));
% training data
xtrain = x(trainingIdx);
ttrain = y(trainingIdx)+noise;
% feature transformation: x... row vector, d... nr of linear basis function dimensions
phi = @(x,d)(ones(d+1,1)*x).^((0:d)'*ones(size(x))); 
% dimensions of linear basis function model
d = 2;
% transformed training data
xtrain_phi = phi(xtrain,d);


%% Task 1.2.2 Optimization: LMS-learning rule vs. closed form
% online LMS-learning rule (true)
wLMS = lmsTrain(xtrain_phi, ttrain, true);
% plot target and regression
figure()
hold on
plot(x,y)
plot(xtrain,ttrain,'*g')
%plot polynom of LMS (w are the coefficients)
plot(x,polyval(fliplr(wLMS),x),'r')
%plot(xtrain,wLMS*xtrain_phi,'r')
legend('original','trainingsset','LMS')