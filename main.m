%% MLVC - Group 12
% Elisabeth Wetzer
% Lena Trautmann
% Hanna Huber

clear all
close all
clc

%% Task 1.1.
% read in data -> new .csv file written by .py-file
data = dlmread('perceptrondataUseful.csv',','); 

% plot the input vectors in R^2
figure()
scatter(data(:,1),data(:,2),20,data(:,3),'filled')
xlabel('x')
ylabel('y')
title('input vectors')

% do feature transformation and plot again
transformed_data = [data(:,1:2).^2 data(:,3)];
figure()
scatter(transformed_data(:,1),transformed_data(:,2),20,transformed_data(:,3),'filled')
xlabel('x')
ylabel('y')
title('transformed input vectors')
%% Task 1.1.2
