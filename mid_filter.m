function data = mid_filter(orig)
%��ֵ�˲�����
%���룺ԭʼ����
%������˲��������
%���ܣ�ʵ����ֵ�˲�
%ע�⣺�������ݵĸ�ʽ�����ݵ������

[row, col] = size(orig);
data = zeros(row, col);
for i=1:col
    temp = orig(:,i);
    data_Medfilt=medfilt1(temp,2);
    data(:,i)=data_Medfilt;
end

