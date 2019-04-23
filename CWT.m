function img = CWT(data, frequency, time, wavename, totalscal, size, limit)
%小波时频图函数
%data:信号 frequency:采样频率 time:采样时间 wavename:小波名称
%totalscal:尺度系数 size:图像尺寸 limit:显示的最大频率
    s = data;
    %原始信号
    t = 0:1/frequency:time;
    %连续小波变换
    Fc = centfrq(wavename);%小波的中心频率
    c = 2*Fc*totalscal;
    scals = c./(1:totalscal);
    f = scal2frq(scals,wavename,1/frequency);%将尺度转化为频率
    coefs = cwt(s,scals,wavename);%求连续小波系数
    imagesc(t,f,abs(coefs));
    set(gca,'YDir','normal');
    colorbar;
    xlabel('时间 t/s');
    ylabel('频率 f/Hz');
    title('小波时频图');
    %ylim([min(f),limit]);
    caxis([0,0.5]);%标准化
    pic = getframe;
    pic.cdata = imresize(pic.cdata,[size,size]);%图片大小
    img = pic.cdata;
    close all;
