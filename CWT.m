function img = CWT(data, frequency, time, wavename, totalscal, size, limit)
%С��ʱƵͼ����
%data:�ź� frequency:����Ƶ�� time:����ʱ�� wavename:С������
%totalscal:�߶�ϵ�� size:ͼ��ߴ� limit:��ʾ�����Ƶ��
    s = data;
    %ԭʼ�ź�
    t = 0:1/frequency:time;
    %����С���任
    Fc = centfrq(wavename);%С��������Ƶ��
    c = 2*Fc*totalscal;
    scals = c./(1:totalscal);
    f = scal2frq(scals,wavename,1/frequency);%���߶�ת��ΪƵ��
    coefs = cwt(s,scals,wavename);%������С��ϵ��
    imagesc(t,f,abs(coefs));
    set(gca,'YDir','normal');
    colorbar;
    xlabel('ʱ�� t/s');
    ylabel('Ƶ�� f/Hz');
    title('С��ʱƵͼ');
    %ylim([min(f),limit]);
    caxis([0,0.5]);%��׼��
    pic = getframe;
    pic.cdata = imresize(pic.cdata,[size,size]);%ͼƬ��С
    img = pic.cdata;
    close all;
