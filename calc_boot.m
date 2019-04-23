function [boot,new_data] = calc_boot(data, window, stride)
%计算窗口滑动的最大次数
% data:数据 window:窗长 stride:步长
data_size = size(data,1);
boot = floor((data_size-window)/stride);
new_data = data(1:(window+boot*stride));
