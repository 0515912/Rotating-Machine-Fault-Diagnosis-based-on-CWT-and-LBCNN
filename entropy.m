function ent = entropy(img)
%信息熵函数
%输入：图片
%输出：信息熵
%功能：提取图片的信息熵
%注意：如果输入的是彩图，会拆成灰度图算均值

row = size(img,1);
col = size(img,2);
temp = zeros(1,256);
result = zeros(size(img,3),1);
for i = 1:size(img,3)
    layer = img(:,:,i);
	for m=1:row  
	    for  n=1:col  
	        if layer(m,n)==0  
	            j=1;  
	        else  
	            j=layer(m,n)+1;  
	        end  
	        temp(j)=temp(j)+1;  
	    end  
	end  
	temp=temp./(row*col);  	   
	for  j=1:length(temp)  
	    if temp(j)==0  
	        result(i) = result(i);  
	    else  
	        result(i) = result(i)-temp(j)*log2(temp(j));  
	    end  
	end  
end

ent = mean(result);