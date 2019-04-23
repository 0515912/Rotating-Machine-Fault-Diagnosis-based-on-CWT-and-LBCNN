function [img] = generate_img(data,img_size,wavename)
%to generate time-frequency map based on original signal
%input: data: original signal dataset
%       img_size: the ideal picture size will be pic_size*pic_size
%output: img

IMG = zeros(img_size,img_size);
[IF,] = wavedec2(data,3,wavename);
points = size(IF,2)-3*img_size^2;
if points < 0
    I_IF = zeros(1,3*img_size^2);
    I_IF(1:size(IF,2)) = IF;
    IF = I_IF;
    disp("add "+num2str(-points)+" points");
else
    disp("discard "+num2str(points)+" points");
end
for i = 1:3
    I_F = IF((i-1)*img_size^2+1:i*img_size^2);
    for j = 1:img_size
        for k = 1:img_size
            IMG(j,k,i)  = (I_F((j-1)*img_size+k)-min(I_F((j-1)*img_size+1:j*img_size)))/(max(I_F((j-1)*img_size+1:j*img_size))-min(I_F((j-1)*img_size+1:j*img_size)));
        end
    end
end
img = im2uint8(IMG);

end

