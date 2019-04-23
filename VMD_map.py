from My_CNN import My_CNN
import matplotlib.pyplot as plt
import csv
import h5py

# 共处理两个数据集，MFPT与澳门大学
global MCU_dir
MCU_dir = 'F:\毕业设计\MCU\振动图\数据记录\\' # 澳门大学数据集保存路径
def load_img():
    '''
    载入图片，只需要做一次
    :return:
    '''
    # MCU
    dir = MCU_dir
    model = My_CNN(64, 17, 1000, dir, channel=1)  # 初始化模型 图片尺寸40 共17类 每类20张图片
    img_dir = 'F:\毕业设计\MCU\振动图\imgs\\'  # 图片保存路径
    model.read_img(img_dir)  # 初次使用需要载入图片

def Lenet_MCU():
    '''
            使用Lenet完整地训练澳门大学数据集，保存最优结果
            :return:
    '''
    dir = MCU_dir  # 数据集保存路径，之后就可以不用重新载入图片
    model = My_CNN(64, 17, 1000, dir, channel=1)  # 初始化模型 图片尺寸40 共17类 每类1000张图片
    model.LeNet_5('Adam', lr=1e-3)
    hist_path = dir + 'LeNet_'
    path = dir + 'LeNet_model.h5'
    model.train_model(1000, 512, model_path=path, hist_path=hist_path, early_save=True, patience=10)

def MCU_output():
    '''
    输出澳门大学混淆矩阵
    :return:
    '''
    dir = MCU_dir
    model = My_CNN(40, 17, 1000, dir, channel=1)  # 初始化模型 图片尺寸40 共16类 每类1000张图片
    model.LBCNN('Adam')
    path = dir + 'LeNet_model.h5'
    model.run_model(path)

if __name__ == '__main__':
    #load_img()
    #Lenet_MCU()
    MCU_output()
