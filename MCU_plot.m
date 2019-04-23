%%
clear all; clc;
load('F:\毕业设计\MCU\MCU_data.mat');
wavename  = 'cmor3-3';
size = 40;
scal = 1024;
frequency = 10240;
time = 1;
depress_pic_path = 'F:\毕业设计\MCU\时频图\imgs\';
for i = 1:17000
        limit = 1000;
        d = Data(:,i);
        img = CWT(d, frequency, time, wavename, scal, size, limit);
        %imwrite(img,[depress_pic_path,int2str(i),'.jpg']);
        disp(int2str(i))
end
