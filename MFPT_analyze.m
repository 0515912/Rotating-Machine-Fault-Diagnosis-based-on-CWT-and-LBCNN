%MFPT ANALYZE
clear all; clc;
load('H:\毕业设计\MFPT\MFPT_data.mat');
load('names_bin.mat');
baseline = data(1:146484*2);%1757808
innerrace = data(146484+1757809:146484*2+1757808);%1025388
outterrace = data(146484+4541005:4541004+146484*2);%1025388
f = [97656,48828];
%1:size(names_bin,1)
%%
% window = 9765;
% img_num = 1000;
% step = 280;
imgsize = 40;
totalscal = 2048;
time = 0.1;
img_path = 'H:\毕业设计\MFPT\时频图\imgs_selection\';
limit = 200;
for i = 74
    wavename =names_bin{i};
    for n = 1:5
        %正常状态
        window = 9765;
        step = 100;
        frequency = f(1);
        data = baseline([1:window]+n*step);
        img = CWT(data, frequency, time, wavename, totalscal, imgsize, limit);
        imwrite(img,[img_path,wavename,'_','baseline',int2str(n),'.jpg']);
        %内圈和外圈
        window = 4880;
        step = 100;
        frequency = f(2);
        data = innerrace([1:window]+n*step);
        img = CWT(data, frequency, time, wavename, totalscal, imgsize, limit);
        imwrite(img,[img_path,wavename,'_','innerrace',int2str(n),'.jpg']);
        data = outterrace([1:window]+n*step);
        img = CWT(data, frequency, time, wavename, totalscal, imgsize, limit);
        imwrite(img,[img_path,wavename,'_','outterrace',int2str(n),'.jpg']);
    end
    disp(wavename);
end

%%
%计算IQA参数
% part1 信息熵
ENT = zeros(imgsize(names_bin,1),1);
for i = 1:imgsize(names_bin,1)
    wavename =names_bin{i};
    ent = 0;
        for n = 1:5
            name = [wavename,'_','baseline',int2str(n),'.jpg'];
            img = imread([img_path,name]);
            ent = ent + entropy(img);
            name = [wavename,'_','innerrace',int2str(n),'.jpg'];
            img = imread([img_path,name]);
            ent = ent + entropy(img);
            name = [wavename,'_','outterrace',int2str(n),'.jpg'];
            img = imread([img_path,name]);
            ent = ent + entropy(img);
        end
    ENT(i) = ent;
    disp(wavename);
end


%%
%计算IQA参数
% part2 Brenner梯度函数
Bre = zeros(imgsize(names_bin,1),1);
for i = 1:imgsize(names_bin,1)
    wavename =names_bin{i};
    bre = 0;
    for m = 0:16
        for n = 1:5
            dir = 'G:\毕业设计\AMU\时频图\imgs_selection\';
            name = [wavename,'_',num2str(m+1),'.',int2str(n),'.jpg'];
            img = imread([dir,name]);
            bre = bre + Brenner(img);
        end
    end
    Bre(i) = bre;
    disp(wavename);
end

%归一化
A = (ENT-min(min(ENT)))/(max(max(ENT))-min(min(ENT)));
B = (Bre-min(min(Bre)))/(max(max(Bre))-min(min(Bre)));
C = (A+B)/2;
plot(C);
xlabel('小波类型');
ylabel('质量指标');