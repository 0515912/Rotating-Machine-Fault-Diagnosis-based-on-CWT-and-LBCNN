function data = mid_filter(orig)
%中值滤波函数
%输入：原始数据
%输出：滤波后的数据
%功能：实现中值滤波
%注意：输入数据的格式是数据点×样本

[row, col] = size(orig);
data = zeros(row, col);
for i=1:col
    temp = orig(:,i);
    data_Medfilt=medfilt1(temp,2);
    data(:,i)=data_Medfilt;
end

