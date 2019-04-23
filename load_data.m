function data= load_data()
%用于载入MFPT数据集
%输出：data: 数据集
root = dir('E:\IMP Lab\毕业设计\一种新型神经网络\MFPT数据集\data\*.mat');% 获取目录
file_name = {root.name}';%  获取文件名
data = [];
for i=1:size(file_name,1)
    disp(['开始载入数据包',num2str(i)]);
    load(strcat('E:\IMP Lab\毕业设计\一种新型神经网络\MFPT数据集\data\' , file_name{i}));
    data = [data;bearing.gs];
end
data = outlier_filter(data);
data = mid_filter(data);



