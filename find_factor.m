function find_factor(n)
%����һ�����֣�����Ļ�������������
m=2;   %��С��������2

disp([num2str(n),'='])

while(1)
    if(~mod(n,m))                                              %�ҵ�������������
        k=m;
        if(n==k)                                               %�ҵ����һ������
            disp(num2str(n))
            break;                                            %����ѭ��
        else
            n=n/k;                                           %��n����������ֵ����ѭ��
            m=1;                                             %��֤�������Ǵ�2��ʼ
            disp([num2str(k),'*'])                  %���ֽ��������ʾ����
        end
    end
    
    m=m+1;                             
end