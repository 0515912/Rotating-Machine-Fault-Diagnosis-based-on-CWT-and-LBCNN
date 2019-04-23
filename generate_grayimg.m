function [img] = generate_grayimg(data,img_size)
%to generate time-frequency map based on original signal
%input: data: original signal dataset
%       img_size: the ideal picture size will be pic_size*pic_size
%output: img

IMG = zeros(img_size,img_size);
for j = 1:img_size
    for k = 1:img_size
        IMG(j,k)  = (data((j-1)*img_size+k)-min(data((j-1)*img_size+1:j*img_size)))/(max(data((j-1)*img_size+1:j*img_size))-min(data((j-1)*img_size+1:j*img_size)));
    end
end
img = im2uint8(IMG);

end

