clc; clear all;
load('MCU_data.mat');
img_path = 'imgs\';
size = 64;
window = 4096;
step = 256;
num=1000;
for i = 1:16
    data = Data(:,(i*1000+1):(i+1)*1000);
    disp(i+1);
    data = reshape(data, 4096000, 1);
    for n = 1:num
    d = data([1:window]+n*step);
    %img = generate_img(d, size, 0);
    img = generate_img(d, size);
    %img = imresize(img, [40, 40]);
    imwrite(img,[img_path,int2str(i*num+n),'.jpg']);
    disp(n);
    end
end
