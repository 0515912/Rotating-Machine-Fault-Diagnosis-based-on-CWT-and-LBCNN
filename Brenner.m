function bre = Brenner(img)
%Brenner�ݶȺ���
%���룺ͼƬ
%�����Brenner�ݶ�
%���ܣ�����ͼ���������
%ע�⣺���������ǲ�ͼ�����ɻҶ�ͼ���ֵ

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