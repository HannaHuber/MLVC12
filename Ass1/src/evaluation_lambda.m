close all
clear all

x = (0:0.1:5);
y = 2*x.^2-12*x+1;
trainingIdx = (1:8:51);
N = length(trainingIdx);
noise = normrnd(0,4,size(trainingIdx));
% training data
xtrain = x(trainingIdx);

% feature transformation: d=8 degree of linear basis function polynom
d=8;
phi = @(x,d)(ones(d+1,1)*x).^((0:d)'*ones(size(x))); 
% transformed training data
xtrain_phi = phi(xtrain,d);


N1=2000;
%choose fixed x=2
xprime = 2;
%exact evaluation of polynom at x=2
yexact = 2*xprime.^2-12*xprime+1;
sum_error=zeros(8,1);
MSE_estimate=zeros(8,1);
sum_ynoisy=zeros(8,1);
bias_estimate=zeros(8,1);
ynoisy_tmp=zeros(N1,8);
var_estimate=zeros(8,1);


for k=1:N1

noise = normrnd(0,4,size(trainingIdx));
% training data

ttrain = y(trainingIdx)+noise;


for lambda_ind=1:8
lambda=exp(lambda_ind-4);

%w=(lambda*I+Phi'*Phi)^(-1)*Phi'*t, as in Bishop, 2006.
w = inv(lambda*eye(9)+xtrain_phi*xtrain_phi')*xtrain_phi*ttrain';
ynoisy = polyval(flipud(w),xprime);

%MSE
error=(ynoisy-yexact)^2;

%Sum mean squared error for all trial runs to estimate mean
sum_error(lambda_ind)=sum_error(lambda_ind)+error;

%Sum y results of model for all trial runs to estimate mean
sum_ynoisy(lambda_ind)=sum_ynoisy(lambda_ind)+ynoisy;

%store y results of model to calculate variance outside of loop
ynoisy_tmp(k,lambda_ind)=ynoisy;

end %for all lambda
end %for all 2000 trials

%Expected Value of MSE
MSE_estimate=sum_error./N1;

%Expected Value of y calculated by Model
expected_y=sum_ynoisy./N1;
bias_estimate=(yexact*ones(8,1)-expected_y).^2;

sum_var=zeros(8,1);
%calculate estimated variance
for p=1:N1
sum_var=sum_var+(ynoisy_tmp(p,:)'-expected_y).^2;
end
var_estimate=sum_var/N1;


figure; plot(1:8,MSE_estimate,'LineWidth',2)
hold on;
plot(1:8,bias_estimate,'LineWidth',2)
plot(1:8,var_estimate,'LineWidth',2)

title('Model Quantities of regularized Model')
legend('MSE Estimate', 'Bias Estimate', 'Variance Estimate')
xlabel('Index of Lambda')
%lambda itself given by exp(index-4)
