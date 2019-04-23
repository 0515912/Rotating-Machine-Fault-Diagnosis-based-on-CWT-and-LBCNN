function bre = Brenner(img)
%Brenner梯度函数
%输入：图片
%输出：Brenner梯度
%功能：评价图像的清晰度
%注意：如果输入的是彩图，会拆成灰度图算均值

img = im2double(img);
row = size(img,1);
col = size(img,2);
result = zeros(size(img,3),1);
for i = 1:size(img,3)
    layer = img(:,:,i);
    temp = 0;
	for m=1:row-2 
	    for  n=1:col  
	        temp = temp + (img(m+2,n,i)-img(m,n,i))^2;  
	    end  
    end  
    result(i) = temp; 
end

bre = mean(result);