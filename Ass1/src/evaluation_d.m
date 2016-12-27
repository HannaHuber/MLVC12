close all
clear all
x = (0:0.1:5);
y = 2*x.^2-12*x+1;
trainingIdx = (1:8:51);
N = length(trainingIdx);
noise = normrnd(0,4,size(trainingIdx));

% training data
xtrain = x(trainingIdx);
%ttrain = y(trainingIdx)+noise;

% feature transformation
phi = @(x,d)(ones(d+1,1)*x).^((0:d)'*ones(size(x))); 


N1=2000; %number of trials
xprime = 2; %exact location of evaluation x=2
yexact = 2*xprime.^2-12*xprime+1;
sum_error=zeros(9,1);
MSE_estimate=zeros(9,1);
sum_ynoisy=zeros(9,1);
bias_estimate=zeros(9,1);
ynoisy_tmp=zeros(N1,9);
var_estimate=zeros(9,1);

for k=1:N1

noise = normrnd(0,4,size(trainingIdx));
% training data
xtrain = x(trainingIdx);
ttrain = y(trainingIdx)+noise;


for d=0:8
    %d= dimensions of linear basis function model
phi = @(x,d)(ones(d+1,1)*x).^((0:d)'*ones(size(x))); 

% transformed training data
xtrain_phi = phi(xtrain,d);

w = pinv(xtrain_phi')*ttrain';

ynoisy = polyval(flipud(w),xprime);

%Mean squared error
error=(ynoisy-yexact)^2;
%Sum mean squared error for all trial runs to estimate mean
sum_error(d+1)=sum_error(d+1)+error;

%Sum y results of model for all trial runs to estimate mean
sum_ynoisy(d+1)=sum_ynoisy(d+1)+ynoisy;

%store y results of model to calculate variance outside of loop
ynoisy_tmp(k,d+1)=ynoisy;

end
end
%Expected Value of MSE
MSE_estimate=sum_error./N1;
%Expected Value of y calculated by Model
expected_y=sum_ynoisy./N1;
% Bias
bias_estimate=(yexact*ones(9,1)-expected_y).^2;

sum_var=zeros(9,1);
%calculate estimated variance
for p=1:N1
sum_var=sum_var+(ynoisy_tmp(p,:)'-expected_y).^2;
end
var_estimate=sum_var/N1;



figure; plot(0:8,MSE_estimate, 'LineWidth',2)
hold on;
plot(0:8,bias_estimate,'LineWidth',2)
plot(0:8,var_estimate,'LineWidth',2)

title('Model quantities of non-regularized Model')
legend('MSE Estimate', 'Bias Estimate', 'Variance Estimate')
xlabel('Dimension d')
