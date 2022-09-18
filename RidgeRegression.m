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
wor = mean(y(1:100,1))

%% Learning weights
w = [0 0 0];
for i = 1:101
    lambda = 0.1*(i-1);
    a = ((xtrain'*xtrain + lambda * eye(max(size(xtrain'*xtrain)))) \ (xtrain'*ytrain))';
    w = [w; a];
end
wr = w(2:end,:);

%% SSE
xtest = [ones(max(size(xtrain)),1) xtest]
wts = [wor*ones(max(size(wr)),1) wr];
SSEr = zeros(max(size(wr)),1);
for i = 1:max(size(wr))
    SSEr(i,1) = sum((wts(i,:)*xtest' - ytest').^2);
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

%% Plot for SSE vs lambda
figure
plot(lambda,SSEr,'o-')
title('sse vs lambda'); xlabel('lambda'); ylabel('sse');