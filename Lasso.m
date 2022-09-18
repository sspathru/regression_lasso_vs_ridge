clc 
clear all
close all
load('data.mat')

%% Standardizing inputs
xtrain = (x(1:100,:) - mean(x(1:100,:)))./std(x(1:100,:));
xtest = (x(101:end,:) - mean(x(1:100,:)))./std(x(1:100,:));

%% Normalizing outputs
ytrain = y(1:100,:) - mean(y(1:100,:));
ytest = y(101:end,:) - mean(y(1:100,:));

%% Determining wo
wol = mean(y(1:100,1))

%% Overall least square estimate
wp = xtrain\ytrain;

%% First value of GE
GE = sign(wp)';

%% Learning weights using TIBSHIRANI algorithm
eps = 10^-4;
w = [0 0 0];
for i = 1:101
    lambda = 0.1*(i-1);
    t = 1/lambda;
        while sum(abs(wp)) > t+eps 
        [wp] = quadprog(2*xtrain'*xtrain,-2*xtrain'*ytrain,GE,t*ones(size(GE,1),1),[],[],[],[],wp);
        GE = [GE;sign(wp)'];
        end
     w = [w; wp'];
end
wts = w(2:end,:);

%% SSE
wo = mean(y(1:100,1));
xtest = [ones(max(size(xtest)),1) xtest]
wts = [wo*ones(max(size(wts)),1) wts];
SSEl = zeros(max(size(wts)),1);
for i = 1:max(size(wts))
    SSEl(i,1) = sum((wts(i,:)*xtest' - ytest').^2);
end

%% Plots for weights vs lambda
lambda = (0:0.1:10)';
figure
plot(lambda,wts(:,2),'o-')
hold on
plot(lambda,wts(:,3),'d-')
hold on
plot(lambda,wts(:,4),'s-')
title('Weights vs lambda'); xlabel('lambda'); ylabel('Weights');
legend('weight1','weight2','weight3','Location','northeastoutside')
hold off

%% Plot for SSE vs Lambda
figure
plot(lambda,SSEl,'o-')
title('SSE vs lambda'); xlabel('lambda'); ylabel('SSE');