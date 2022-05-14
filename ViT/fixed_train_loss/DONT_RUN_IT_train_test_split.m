
datainput = load('D:\ERG4901\ViT_Project\Input.mat');
dataoutput = load('D:\ERG4901\ViT_Project\Output.mat');
% Input
x = datainput.Input;
% Out
y = dataoutput.QnV;
randpm = randperm(12500);
x_pm = x(randpm,:,:,:);
y_pm = y(randpm,:,:,:);
xtrain = x_pm(1:10000,:,:,:);
xtest = x_pm(10001:end,:,:,:);
ytrain = y_pm(1:10000,:);
ytest = y_pm(10001:end,:);
%%%%% q,v
% TrainX = struct('Input',xtrain);
% TestX = struct('Input',xtest);
% TrainY = struct('QnV', ytrain);
% TestY = struct('QnV', ytest);
save('TrainXdata.mat','xtrain', '-v7.3')
save('TestXdata.mat','xtest', '-v7.3')
save('TrainYdata.mat','ytrain', '-v7.3')
save('TestYdata.mat','ytest', '-v7.3')
