function data= load_data()
%��������MFPT���ݼ�
%�����data: ���ݼ�
root = dir('E:\IMP Lab\��ҵ���\һ������������\MFPT���ݼ�\data\*.mat');% ��ȡĿ¼
file_name = {root.name}';%  ��ȡ�ļ���
data = [];
for i=1:size(file_name,1)
    disp(['��ʼ�������ݰ�',num2str(i)]);
    load(strcat('E:\IMP Lab\��ҵ���\һ������������\MFPT���ݼ�\data\' , file_name{i}));
    data = [data;bearing.gs];
end
data = outlier_filter(data);
data = mid_filter(data);



