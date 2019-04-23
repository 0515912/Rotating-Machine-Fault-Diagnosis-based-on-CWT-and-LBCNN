%AMU ANALYZE
clear all; clc;
load('F:\毕业设计\MCU\MCU_data.mat');
load('names_bin.mat'); %载入小波名称
%%
%绘制小波时频图
dia = 64;
totalscal = 1024;
frequency = 10240;
time = 0.4;
depress_pic_path = 'G:\毕业设计\MCU\时频图\imgs_selection\';

for i = 1:size(names_bin,1)
    for m = 0:16
        for n = 1:5
            data = Data(:,n+m*1000);
            wavename =names_bin{i};
            limit = 150;
            img = CWT(data, frequency, time, wavename, totalscal, dia, limit);
            imwrite(img,[depress_pic_path,wavename,'_',int2str(m+1),'.',int2str(n),'.jpg']);
            disp(wavename);
        end
    end
end

%%
%计算IQA参数
% part1 信息熵
ENT = zeros(size(names_bin,1),1);
for i = 1:size(names_bin,1)
    wavename =names_bin{i};
    ent = 0;
    for m = 0:16
        for n = 1:5
            dir = 'G:\毕业设计\AMU\时频图\imgs_selection\';
            name = [wavename,'_',num2str(m+1),'.',int2str(n),'.jpg'];
            img = imread([dir,name]);
            ent = ent + entropy(img);
        end
    end
    ENT(i) = ent;
    disp(wavename);
end


%%
%计算IQA参数
% part2 Brenner梯度函数
Bre = zeros(size(names_bin,1),1);
for i = 1:size(names_bin,1)
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
