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
figure(1)
scatter(data(:,1),data(:,2),20,data(:,3),'filled')
xlabel('x')
ylabel('y')
title('input vectors')
% save the figure
set(1,'Units','Inches');
pos = get(1,'Position');
set(1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(1,'figures/inputVectors','-dpdf','-r0')


% do feature transformation and plot again
transformed_data = [data(:,1:2).^2 data(:,3)];
figure(2)
scatter(transformed_data(:,1),transformed_data(:,2),20,transformed_data(:,3),'filled')
xlabel('x')
ylabel('y')
title('transformed input vectors')
% save the figure
set(2,'Units','Inches');
pos = get(2,'Position');
set(2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(2,'figures/transformedInputVectors','-dpdf','-r0')


%% Task 1.1.2
