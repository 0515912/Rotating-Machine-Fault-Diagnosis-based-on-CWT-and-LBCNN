clear all; clc;
load MFPT_data.mat;
baseline = data(1:1757808);%1757808
innerrace = data(1757809:2783196);%1025388
outterrace = data(4541005:end);%1025388
outterrace1 = data(2783197:4541004); %1757808
f = [97656,48828];
imgsize = 40;
totalscal = 2048;
time = 0.1;
img_path = 'imgs\';
limit = 200;
counter = 1;
wavename = 'cmor3-3';
num = 1000;

for n = 1:num
    %正常状态
    window = 9765;
    step = 100;
    frequency = f(1);
    data = baseline([1:window]+n*step);
    img = CWT(data, frequency, time, wavename, totalscal, imgsize, limit);
    imwrite(img,[img_path,int2str(counter),'.jpg']);
    counter = counter+1;
end

for i = 0:6
    for n = [1:num]+i*1417
        %内圈
        window = 4880;
        step = 100;
        frequency = f(2);
        data = innerrace([1:window]+n*step);
        img = CWT(data, frequency, time, wavename, totalscal, imgsize, limit);
        imwrite(img,[img_path,int2str(counter),'.jpg']);
        counter = counter+1;
    end
end

for i = 0:6
    for n = [1:num]+i*1417
        %外圈
        data = outterrace([1:window]+n*step);
        img = CWT(data, frequency, time, wavename, totalscal, imgsize, limit);
        imwrite(img,[img_path,int2str(counter),'.jpg']);
        counter = counter+1;
    end
end
for n = 1:num
    %外圈1
    window = 9765;
    step = 100;
    frequency = f(1);
    data = outterrace1([1:window]+n*step);
    img = CWT(data, frequency, time, wavename, totalscal, imgsize, limit);
    imwrite(img,[img_path,int2str(counter),'.jpg']);
    counter = counter+1;
end

wav='coif1';%小波名称
[phi,psi,xval] = wavefun(wav); 
plot(xval,psi);

plot(baseline(1:1000))
pic = getframe;
img = pic.cdata;
dir = 'coif1.jpg';
imwrite(img,dir)
