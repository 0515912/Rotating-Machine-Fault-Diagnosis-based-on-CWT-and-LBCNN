function result = get_right_shape(data, number)
remain = rem(size(data,1),number);
result = data(1:(end-remain));