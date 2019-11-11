%% 画图
%% optimizer
data1 = csvread('Adam.csv');
data2 = csvread('SGD.csv');
data3 = csvread('Adadelta.csv');
data4 = csvread('Adagrad.csv');
plot(data1(1,:)*100,'Linewidth', 2)
hold on;
plot(data2(1,:)*100,'Linewidth', 2)
hold on;
plot(data3(1,:)*100,'Linewidth', 2)
hold on;
plot(data4(1,:)*100,'Linewidth', 2)
legend('Adam','SGD','Adadelta','Adagrad');
xlabel('Epoch')
ylabel('Accuracy(%)')

%% lenet
data1 = csvread('RLBCNN.csv');
data2 = csvread('RLenet.csv');
plot(data1(1,:)*100,'Linewidth', 2)
hold on;
plot(data2(1,:)*100,'Linewidth', 2)
legend('Proposed architecture','LeNet-5');
xlabel('Epoch')
ylabel('Accuracy(%)')


%% early stopping
data = csvread('LBCNN.csv');
figure;
plot(data(1,:)*100,'Linewidth',2)
hold on;
plot(data(2,:)*100,'Linewidth',2)
legend('Train accuracy','Validation accuracy');
xlabel('Epoch')
ylabel('Accuracy(%)')

figure;
plot(data(3,:),'y','Linewidth',2)
hold on;
plot(data(4,:),'b','Linewidth',2)
hold on;
plot(125,data(4,125),'r.','Linewidth',5)
hold on;
plot(135,data(4,135),'g.','Linewidth',5)
legend('Train loss','Validation loss');
xlabel('Epoch')
ylabel('Loss')


%% 
data = csvread('Lenet.csv');
min(data(4,:))

%% AMU
%% 画图
%% optimizer
data1 = csvread('Adam.csv');
data2 = csvread('SGD.csv');
data3 = csvread('Adadelta.csv');
data4 = csvread('Adagrad.csv');
plot(data1(1,:)*100,'Linewidth', 2)
hold on;
plot(data2(1,:)*100,'Linewidth', 2)
hold on;
plot(data3(1,:)*100,'Linewidth', 2)
hold on;
plot(data4(1,:)*100,'Linewidth', 2)
legend('Adam','SGD','Adadelta','Adagrad');
xlabel('Epoch')
ylabel('Accuracy(%)')

%% lenet
data1 = csvread('RLBCNN.csv');
data2 = csvread('RLenet.csv');
plot(data1(1,:)*100,'Linewidth', 2)
hold on;
plot(data2(1,:)*100,'Linewidth', 2)
legend('Proposed architecture','LeNet-5');
xlabel('Epoch')
ylabel('Accuracy(%)')


%% early stopping
data = csvread('LBCNN.csv');
figure;
plot(data(1,:)*100,'Linewidth',2)
hold on;
plot(data(2,:)*100,'Linewidth',2)
legend('Train accuracy','Validation accuracy');
xlabel('Epoch')
ylabel('Accuracy(%)')

figure;
plot(data(3,:),'y','Linewidth',2)
hold on;
plot(data(4,:),'b','Linewidth',2)
hold on;
plot(236,data(4,236),'r.','Linewidth',5)
hold on;
plot(336,data(4,336),'gx','Linewidth',5)
legend('Train loss','Validation loss');
xlabel('Epoch')
ylabel('Loss')


%% 
data = csvread('Lenet.csv');
min(data(4,:))

