function [boot,new_data] = calc_boot(data, window, stride)
%���㴰�ڻ�����������
% data:���� window:���� stride:����
data_size = size(data,1);
boot = floor((data_size-window)/stride);
new_data = data(1:(window+boot*stride));
