function ent = entropy(img)
%��Ϣ�غ���
%���룺ͼƬ
%�������Ϣ��
%���ܣ���ȡͼƬ����Ϣ��
%ע�⣺���������ǲ�ͼ�����ɻҶ�ͼ���ֵ

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