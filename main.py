# main function for undergraduate thesis
# created by Matthew Lin
# data: 10/03/2019

from My_CNN import My_CNN
import matplotlib.pyplot as plt
import csv
import h5py


global MFPT_dir, MCU_dir
MFPT_dir =  '数据记录\\'
MCU_dir = '数据记录\\'


def load_img():
    '''
    载入图片，只需要做一次
    :return:
    '''
    # MFPT
    dir = MFPT_dir
    model = My_CNN(40, 16, 1000, dir)  # 初始化模型
    img_dir = 'imgs\\'   # 图片保存路径
    model.read_img(img_dir)

    # MCU
    dir = MCU_dir
    model = My_CNN(40, 17, 1000, dir)  # 初始化模型
    img_dir = 'F:\毕业设计\MCU\时频图\imgs\\'  # 图片保存路径
    model.read_img(img_dir)

def OP_MFPT():
    '''
    对数据集，比较不同优化器下的LBCNN网络的效果
    :return: 模型会自动保存为H5格式，训练历史数据会自动保存为二进制格式
    '''
    dir = MFPT_dir
    model = My_CNN(40, 16, 1000, dir)  # 初始化模型

    # Adam
    model.LBCNN('Adam')
    hist_path = dir + 'Adam_'
    path = dir + 'Adam_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path) # 迭代次数为30次

    # SGD
    model.LBCNN('SGD')
    hist_path = dir + 'SGD_'
    path = dir + 'SGD_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path)

    # Adagrad
    model.LBCNN('Adagrad')
    hist_path = dir + 'Adagrad_'
    path = dir + 'Adagrad_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path)

    # Adadelta
    model.LBCNN('Adadelta')
    hist_path = dir + 'Adadelta_'
    path = dir + 'Adadelta_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path)


def OP_MCU():
    '''
        对数据集，比较不同优化器下的LBCNN网络的效果
        :return: 模型会自动保存为H5格式，训练历史数据会自动保存为二进制格式
    '''
    dir = MCU_dir
    model = My_CNN(40, 17, 1000, dir)  # 初始化模型

    # Adam
    model.LBCNN('Adam')
    hist_path = dir + 'Adam_'
    path = dir + 'Adam_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path)

    # SGD
    model.LBCNN('SGD')
    hist_path = dir + 'SGD_'
    path = dir + 'SGD_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path)

    # Adagrad
    model.LBCNN('Adagrad')
    hist_path = dir + 'Adagrad_'
    path = dir + 'Adagrad_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path)

    # Adadelta
    model.LBCNN('Adadelta')
    hist_path = dir + 'Adadelta_'
    path = dir + 'Adadelta_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path)


def LBCNN_MFPT():
    '''
    使用LBCNN完整地训练数据集，保存最优结果
    :return:
    '''
    dir = MFPT_dir
    model = My_CNN(40, 16, 1000, dir)  # 初始化模型
    model.LBCNN('Adam')
    hist_path = dir + 'LBCNN_'
    path = dir + 'LBCNN_model.h5'
    model.train_model(1000, 512, model_path=path, hist_path=hist_path, early_save=True)

def LBCNN_MCU():
    '''
        使用LBCNN完整地训练数据集，保存最优结果
        :return:
    '''
    dir = MCU_dir
    model = My_CNN(40, 17, 1000, dir)  # 初始化模型
    model.LBCNN('Adam', lr=1e-4)
    hist_path = dir + 'LBCNN_'
    path = dir + 'LBCNN_model.h5'
    model.train_model(1000, 512, model_path=path, hist_path=hist_path, early_save=True, patience=100)


def Lenet_MFPT():
    '''
        使用Lenet完整地训练数据集，保存最优结果
        :return:
    '''
    dir = MFPT_dir
    model = My_CNN(40, 16, 1000, dir)  # 初始化模型
    model.LeNet_5('Adam', lr=1e-3)
    hist_path = dir + 'LeNet_'
    path = dir + 'LeNet_model.h5'
    model.train_model(1000, 512, model_path=path, hist_path=hist_path, early_save=True)

def Lenet_MCU():
    '''
            使用Lenet完整地训练数据集，保存最优结果
            :return:
    '''
    dir = MCU_dir  # 数据集保存路径
    model = My_CNN(40, 17, 1000, dir)  # 初始化模型
    model.LeNet_5('Adam', lr=1e-4)
    hist_path = dir + 'LeNet_'
    path = dir + 'LeNet_model.h5'
    model.train_model(1000, 512, model_path=path, hist_path=hist_path, early_save=True, patience=100)




def plot():
    '''
    画图
    :return:
    '''
    dir = MCU_dir  # 数据集保存路径
    model = My_CNN(40, 17, 1000, dir)  # 初始化模型
    hist_path = dir + 'Adam_'
    history = model.load_data(hist_path, 'history')
    plt.figure()
    plt.plot(history['acc'])
    hist_path = dir + 'SGD_'
    history = model.load_data(hist_path, 'history')
    plt.plot(history['acc'])
    hist_path = dir + 'Adadelta_'
    history = model.load_data(hist_path, 'history')
    plt.plot(history['acc'])
    hist_path = dir + 'Adagrad_'
    history = model.load_data(hist_path, 'history')
    plt.plot(history['acc'])
    plt.legend(['Adam', 'SGD', 'Adadelta', 'Adagrad'])
    plt.show()

def csv_trans(file, path):
    '''
    用于将二进制文件变为csv的可读格式
    :param file:
    :param path:
    :return:
    '''
    with open(path, "w", newline='') as f:
        # with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerows([file['acc']])
        writer.writerows([file['val_acc']])
        writer.writerows([file['loss']])
        writer.writerows([file['val_loss']])
        # writer.writerows([file['time']]) # 训练时间未展示
        f.close()

def write():
    # 将二进制文件写为csv格式，方便之后用MATLAB画图
    # MFPT
    dir = MFPT_dir  # 数据集保存路径
    model = My_CNN(40, 16, 1000, dir)  # 初始化模型

    # LBCNN+Adam
    hist_path = dir + 'Adam_'
    history = model.load_data(hist_path, 'history')
    filename = dir+'Adam.csv'
    csv_trans(history, filename)

    # LBCNN+SGD
    hist_path = dir + 'SGD_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'SGD.csv'
    csv_trans(history, filename)

    # LBCNN+Adagrad
    hist_path = dir + 'Adagrad_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'Adagrad.csv'
    csv_trans(history, filename)

    # LBCNN+Adadelta
    hist_path = dir + 'Adadelta_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'Adadelta.csv'
    csv_trans(history, filename)

    #LBCNN
    hist_path = dir + 'LBCNN_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'LBCNN.csv'
    csv_trans(history, filename)

    # Lenet
    hist_path = dir + 'Lenet_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'Lenet.csv'
    csv_trans(history, filename)


    # race
    hist_path = dir + 'RLenet_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'RLenet.csv'
    csv_trans(history, filename)
    hist_path = dir + 'RLBCNN_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'RLBCNN.csv'
    csv_trans(history, filename)

    # MCU
    dir = MCU_dir  # 数据集保存路径
    model = My_CNN(40, 17, 1000, dir)  # 初始化模型
    # LBCNN+Adam
    hist_path = dir + 'Adam_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'Adam.csv'
    csv_trans(history, filename)
    # LBCNN+SGD
    hist_path = dir + 'SGD_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'SGD.csv'
    csv_trans(history, filename)
    # LBCNN+Adagrad
    hist_path = dir + 'Adagrad_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'Adagrad.csv'
    csv_trans(history, filename)
    # LBCNN+Adadelta
    hist_path = dir + 'Adadelta_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'Adadelta.csv'
    csv_trans(history, filename)
    # LBCNN
    hist_path = dir + 'LBCNN_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'LBCNN.csv'
    csv_trans(history, filename)
    # Lenet
    hist_path = dir + 'Lenet_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'Lenet.csv'
    csv_trans(history, filename)
    # race
    hist_path = dir + 'RLenet_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'RLenet.csv'
    csv_trans(history, filename)
    hist_path = dir + 'RLBCNN_'
    history = model.load_data(hist_path, 'history')
    filename = dir + 'RLBCNN.csv'
    csv_trans(history, filename)

def MFPT_race():
    '''
    比较LBCNN和Lenet在数据集中的表现
    :return:
    '''
    dir = MFPT_dir  # 数据集保存路径
    model = My_CNN(40, 16, 1000, dir)  # 初始化模型
    model.LeNet_5('Adam', lr=1e-3)
    hist_path = dir + 'RLeNet_'
    path = dir + 'RLeNet_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path, early_save=False)
    model.LBCNN('Adam', lr=1e-3)
    hist_path = dir + 'RLBCNN_'
    path = dir + 'RLBCNN_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path, early_save=False)

def MCU_race():
    '''
        比较LBCNN和Lenet在数据集中的表现
        :return:
    '''
    dir = MCU_dir  # 数据集保存路径
    model = My_CNN(40, 17, 1000, dir)  # 初始化模型
    model.LeNet_5('Adam', lr=1e-4)
    hist_path = dir + 'RLeNet_'
    path = dir + 'RLeNet_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path, early_save=False)
    model.LBCNN('Adam', lr=1e-4)
    hist_path = dir + 'RLBCNN_'
    path = dir + 'RLBCNN_model.h5'
    model.train_model(30, 512, model_path=path, hist_path=hist_path, early_save=False)

def replacement(path):
    '''
    :param path: 模型路径
    :return:
    '''
    f = h5py.File(path, 'r')
    a = f.attrs.get('model_config')
    b = a.decode('UTF-8')
    ori_str = '{\"name\": \"lbc_1\", \"trainable\": true}}'
    new_str = '{\"name\": \"lbc_1\", \"trainable\": true, \"filters\": 120, \"kernel_size\": 5}}'
    b = b.replace(ori_str, new_str)
    c = b.encode('utf-8')
    f.close()
    f = h5py.File(path, 'a')
    f.attrs.modify('model_config', c)
    f.close()

def MFPT_output():
    '''
    输出混淆矩阵
    :return:
    '''
    dir = MFPT_dir
    model = My_CNN(40, 16, 1000, dir)  # 初始化模型
    model.LBCNN('Adam')
    path = dir + 'LBCNN_model.h5'
    model.run_model(path)


def MCU_output():
    '''
    输出混淆矩阵
    :return:
    '''
    dir = MCU_dir
    model = My_CNN(40, 17, 1000, dir)  # 初始化模型
    model.LBCNN('Adam')
    path = dir + 'LBCNN_model.h5'
    model.run_model(path)


if __name__ == '__main__':
   OP_MFPT()
   OP_MCU()
   LBCNN_MFPT()
   LBCNN_MCU()
   Lenet_MFPT()
   Lenet_MCU()
   plot()
   write()
   MFPT_race()
   MCU_race()
   path = []
   replacement(path)
   MFPT_output()
   MCU_output()


