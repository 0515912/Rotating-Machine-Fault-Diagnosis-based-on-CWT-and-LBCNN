function find_factor(n)
%输入一个数字，在屏幕上输出它的因数
m=2;   %最小的质数是2

disp([num2str(n),'='])

while(1)
    if(~mod(n,m))                                              %找到可以整除的数
        k=m;
        if(n==k)                                               %找到最后一个质数
            disp(num2str(n))
            break;                                            %跳出循环
        else
            n=n/k;                                           %将n除以质数的值继续循环
            m=1;                                             %保证质数还是从2开始
            disp([num2str(k),'*'])                  %将分解的质数显示出来
        end
    end
    
    m=m+1;                             
end