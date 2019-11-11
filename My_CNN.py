# This is a CNN model for fault diagnosis
# author: Matthew Lin
# date: 05/03/2019
# version: 0.0.1

import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
from keras.optimizers import Adam, SGD, Adadelta, Adagrad
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split  # 用于分割数据集
from xlrd import open_workbook
from xlutils.copy import copy
import matplotlib.image as mpimg
import pickle
import sys
import os
import time
from LBCNN import LBC
from LBC import LBC as LB
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '' # 强制keras使用CPU训练网络，防止GPU过载造成蓝屏

class My_CNN(object):
    '''
    My_CNN is designed to facilitate the speed of CNN construction
    '''
    def __init__(self, img_size, class_num, class_vol, dir, channel=3):
        '''
        初始化
        :param img_size: 图片的尺寸，如32*32的图片的size就是32
        :param class_num: 分类的个数
        :param class_vol: 每类的数据集个数
        :param dir: 数据集保存路径，由图片生成的训练集、测试集、验证集，训练历史记录、模型等都保存在这里
        :param channel: 图片的通道数，彩图为3，灰度图为1，默认为3
        '''
        print('欢迎使用My_CNN...')
        self.img_size = img_size
        self.class_num = class_num
        self.class_vol = class_vol
        self.train_data = []
        self.test_data = []
        self.vali_data = []
        self.train_label = []
        self.test_label = []
        self.model = []
        self.time = []
        self.dir = dir
        self.channel = channel
        print('图片尺寸: ', str(img_size))
        print('分类个数: ', str(class_num))
        print('图片总数: ', str(class_num*class_vol))
        print('数据集保存位置: ', dir)
        print('图片通道数: ', channel)
        # 生成标签
        self.generate_label()


    @staticmethod
    def excel_write(row, column, input, xls_dir):
       '''
       将数据写入excel，速度较慢，但可读性好
       :param row: 写入行
       :param column: 写入列
       :param input: 写入内容
       :param xlsdir: 表格路径
       :return:
       '''
       workbook = open_workbook(xls_dir)
       new_workbook = copy(workbook)
       worksheet = new_workbook.get_sheet(0)  # get the first sheet
       worksheet.write(row, column, input)  # input data
       new_workbook.save(xls_dir)  # save the workbook

    @staticmethod
    def save_data(data, dir, name):
        '''
        以二进制形式保存数据集，速度快，不可直接查看
        :param data: 要保存的数据集
        :param dir: 保存路径
        :param name: 保存名称
        '''
        with open(dir + name + u'.txt', 'wb+') as file:
              pickle.dump(data, file)

    @staticmethod
    def load_data(dir, name):
        '''
        载入二进制数据集
        :param dir:保存路径
        :param name: 保存名称
        :return: 数据集
        '''
        with open(dir + name + u'.txt', 'rb+') as file:
             return pickle.load(file)

    @staticmethod
    def find_max(arr):
        '''
        寻找并返回数组中最大值的序号
        :param arr: 数组
        :return:
        '''
        for i in range(len(arr)):
            if arr[i] == max(arr):
                break
        return i


    def read_img(self, dir):
        '''
        读取图片，划分数据集，将数据集以二进制的格式保存起来，方便以后使用
        对于一个数据集，只需要运行一次
        :param dir: 图片路径
        :return: 测试集，训练集，验证集，并保存到相应路径中
        '''
        save_dir = self.dir
        print('正在读入图片...')
        size = self.img_size*1
        vol = self.class_vol*1
        num = self.class_num*1
        train_data = self.train_data
        test_data = self.test_data
        vali_data = self.vali_data
        gross = num*vol
        temp = []
        for i in range(gross):
            name = (dir + str(i + 1) + u'.jpg')
            img = mpimg.imread(name)
            img = np.array(img, dtype='float') / 255.
            temp.append(img)
            if (i+1) == vol:
                print('当前类别: ' + str((i+1) // vol) + '/' + str(num))
                x_test, x_train_vali = train_test_split(temp, test_size=0.9, random_state=0)
                test_data.append(x_test)
                x_train, x_vali = train_test_split(x_train_vali, test_size=1 / 9, random_state=0)
                train_data.append(x_train)
                vali_data.append(x_vali)
                test_data = np.reshape(test_data, [-1, size, size, self.channel])
                train_data = np.reshape(train_data, [-1, size, size, self.channel])
                vali_data = np.reshape(vali_data, [-1, size, size, self.channel])
                temp = []
            elif (i+1) % vol == 0 and i > vol:
                print('当前类别: ' + str((i+1) // vol) + '/' + str(num))
                x_test_temp, x_train_vali = train_test_split(temp, test_size=0.9, random_state=0)
                x_train_temp, x_vali_temp = train_test_split(x_train_vali, test_size=1 / 9, random_state=0)
                x_test_temp = np.reshape(x_test_temp, [-1, size, size, self.channel])
                x_train_temp = np.reshape(x_train_temp, [-1, size, size, self.channel])
                x_vali_temp = np.reshape(x_vali_temp, [-1, size, size, self.channel])
                test_data = np.vstack((test_data, x_test_temp))
                train_data = np.vstack((train_data, x_train_temp))
                vali_data = np.vstack((vali_data, x_vali_temp))
                temp = []
        print('图片读入完成，正在保存数据集...')
        My_CNN.save_data(train_data, save_dir, 'train_data')
        print('训练集已保存！')
        My_CNN.save_data(test_data, save_dir, 'test_data')
        print('测试集已保存！')
        My_CNN.save_data(vali_data, save_dir, 'vali_data')
        print('验证集已保存！')
        sys.exit(0)

    def generate_label(self):
        '''
        生成训练集和测试集的标签，验证集的标签和测试集相同
        :return: 训练集标签，测试集标签
        '''
        num = self.class_num*1
        train_vol = int(self.class_vol*0.8)
        test_vol = int(self.class_vol*0.1)
        print('正在生成训练集标签...')
        train_gross = num*train_vol
        train_label = [int(np.ceil((x+1)/train_vol)-1) for x in range(train_gross)]
        self.train_label = to_categorical(train_label)
        print('正在生成测试集标签...')
        test_gross = num * test_vol
        test_label = [int(np.ceil((x + 1) / test_vol) - 1) for x in range(test_gross)]
        self.test_label = to_categorical(test_label)

    @staticmethod
    def select_op(op, learning_rate):
        '''
        选择优化器
        :param op:优化器名称，共有四种可以选择
        :return: 优化器
        '''
        print('正在选择优化器...')
        print('当前学习率; ', learning_rate)
        if op == 'Adam':
            print('选择Adam优化方法')
            optimiser = Adam(lr=learning_rate)
        elif op == 'SGD':
            print('选择SGD优化方法')
            optimiser = SGD(lr=learning_rate, nesterov=True)
        elif op == 'Adagrad':
            print('选择Adagrad优化方法')
            optimiser = Adagrad(lr=learning_rate)
        elif op == 'Adadelta':
            print('选择Adadelta优化方法')
            optimiser = Adadelta(lr=learning_rate)
        else:
            print('My_CNN不支持这种优化器！')
            sys.exit(1)
        return optimiser

    def LeNet_5(self, op, lr=1e-3):
        '''
        搭建LeNet-5结构
        :param op: 优化器名称
        :param lr: 学习率，默认为0.001
        :return: LeNet-5模型
        '''
        model = Sequential()
        print('正在搭建LeNet-5结构...')
        size = self.img_size*1
        # -----第一卷积层-----
        conv1 = Convolution2D(filters=6, kernel_size=(5, 5), input_shape=(size, size, self.channel))
        model.add(conv1)
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        pool1 = MaxPool2D(pool_size=(2, 2))
        model.add(pool1)

        # -----第二卷积层-----
        conv2 = Convolution2D(filters=16, kernel_size=(5, 5))
        model.add(conv2)
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        pool2 = MaxPool2D(pool_size=(2, 2))
        model.add(pool2)

        # -----第三卷积层-----
        conv3 = Convolution2D(filters=120, kernel_size=(5, 5))
        model.add(conv3)
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        # -----第一全连接层-----
        model.add(Flatten())
        dense1 = Dense(size*size)
        model.add(dense1)
        model.add(Activation('relu'))

        # -----第二全连接层-----
        dense2 = Dense(self.class_num)  # 分类
        model.add(dense2)
        model.add(Activation('softmax'))
        model.add(Activation('relu'))

        # 选择优化器
        Op = My_CNN.select_op(op, lr)
        model.compile(optimizer=Op, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    def LBCNN(self, op, lr=1e-3):
        '''
        搭建LBCNN结构
        :param op: 优化器类型
        :param lr: 学习率，默认为0.001
        :return: LBCNN模型
        '''

        model = Sequential()
        print('正在搭建LBCNN结构...')
        size = self.img_size

        # -----第一卷积层-----
        conv1 = Convolution2D(filters=6, kernel_size=5, input_shape=[size, size, self.channel])
        model.add(conv1)
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        pool1 = AveragePooling2D(pool_size=(2, 2))
        model.add(pool1)

        # -----第二卷积层-----
        conv2 = Convolution2D(filters=16, kernel_size=5)
        model.add(conv2)
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        pool2 = AveragePooling2D(pool_size=(2, 2))
        model.add(pool2)

        # -----局部二值卷积层-----
        # 本层由一个LBP mask层和一个卷积层组成
        # lbc = LB(filters=120, kernel_size=5, sparsity=0.9)
        # lbc = LBC(filters=120, kernel_size=(5, 5), sparsity=0.9) # 稀疏度取为0.9
        model.add(lbc)
        model.add(Activation('relu')) # 用relu加上非线性成分
        conv3 = Convolution2D(filters=120, kernel_size=1) # 不改变特征图大小，卷积层的过滤器大小为1
        model.add(conv3)
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        # -----第一全连接层-----
        model.add(Flatten())
        dense1 = Dense(size*size)
        model.add(dense1)
        # model.add(BatchNormalization())
        model.add(Activation('relu'))

        # -----第二全连接层-----
        dense2 = Dense(self.class_num)  # 分类
        model.add(dense2)
        model.add(Activation('softmax'))

        # 选择优化器
        Op = My_CNN.select_op(op, lr)
        model.compile(optimizer=Op, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    def layer_output(self, model_path, save_path):
        '''
        生成中间层输出
        :param model_path: 模型保存路径
        :param save_path: 输出保存路径
        :return:
        '''
        print('正在读入数据集...')
        train_data = My_CNN.load_data(self.dir, 'train_data')
        print('训练集已读入！')
        print('训练集图片共', str(train_data.shape[0]), '张')
        print('正在打乱数据集顺序...')
        train_index = np.arange(train_data.shape[0])
        np.random.shuffle(train_index)
        train_data = train_data[train_index, :, :, :]
        train_label = self.train_label[train_index, :]
        print('当前模型: ', model_path)

        print('正在载入网络...')
        model = load_model(model_path, custom_objects={'LBC': LBC})
        name = []
        for layer in model.layers:
            name.append(layer.name)
            print('第', str(len(name)), '层: ', layer.name)
        pool1 = Model(inputs=model.input, outputs=model.get_layer(name[2]).output)
        pool2 = Model(inputs=model.input, outputs=model.get_layer(name[5]).output)
        dense1 = Model(inputs=model.input, outputs=model.get_layer(name[9]).output)
        dense2 = Model(inputs=model.input, outputs=model.get_layer(name[11]).output)
        lbc = Model(inputs=model.input, outputs=model.get_layer(name[6]).output)
        sm = Model(inputs=model.input, outputs=model.get_layer(name[12]).output)
        w = lbc.get_weights()
        out = np.reshape(sm.predict(train_data), [train_data.shape[0], -1])

        # 训练集输出
        print('正在保存训练集输出...')
        pool1_out = np.reshape(pool1.predict(train_data), [train_data.shape[0], -1])
        pool2_out = np.reshape(pool2.predict(train_data), [train_data.shape[0], -1])
        dense1_out = np.reshape(dense1.predict(train_data), [train_data.shape[0], -1])
        dense2_out = np.reshape(dense2.predict(train_data), [train_data.shape[0], -1])
        My_CNN.save_data(pool1_out, save_path, 'train_pool1')
        My_CNN.save_data(pool2_out, save_path, 'train_pool2')
        My_CNN.save_data(dense1_out, save_path, 'train_dense1')
        My_CNN.save_data(dense2_out, save_path, 'train_dense2')

        print('正在读入数据集...')
        test_data = My_CNN.load_data(self.dir, 'test_data')
        print('测试集已读入！')
        print('测试集图片共', str(test_data.shape[0]), '张')
        print('正在打乱数据集顺序...')
        test_index = np.arange(test_data.shape[0])
        np.random.shuffle(test_index)
        test_data = test_data[test_index, :, :, :]
        test_label = self.test_label[test_index, :]

        # 测试集输出
        print('正在保存测试集输出...')
        pool1_out = np.reshape(pool1.predict(test_data), [test_data.shape[0], -1])
        pool2_out = np.reshape(pool2.predict(test_data), [test_data.shape[0], -1])
        dense1_out = np.reshape(dense1.predict(test_data), [test_data.shape[0], -1])
        dense2_out = np.reshape(dense2.predict(test_data), [test_data.shape[0], -1])
        My_CNN.save_data(pool1_out, save_path, 'test_pool1')
        My_CNN.save_data(pool2_out, save_path, 'test_pool2')
        My_CNN.save_data(dense1_out, save_path, 'test_dense1')
        My_CNN.save_data(dense2_out, save_path, 'test_dense2')

        # 保存标签
        print('正在保存标签...')
        My_CNN.save_data(train_label, save_path, 'train_label')
        My_CNN.save_data(test_label, save_path, 'test_label')



    def train_model(self, epoch, batch, hist_path, auto_save=True, model_path='‪C:\\Users\Public\Desktop\\', early_save=False, patience=10):
        '''
        训练网络
        :param epoch: 网络迭代次数
        :param batch: 一次输入网络的图片数
        :param hist_path: loss和accuracy的保存位置
        :param auto_save: 是否自动保存模型，默认打开
        :param model_path: 模型的保存位置
        :param early_save: 是否提前终止，默认关闭
        :param patience: 提前终止的容忍度，默认为10
        :return: 无
        '''
        print('正在读入数据集...')
        train_data = My_CNN.load_data(self.dir, 'train_data')
        print('训练集已读入！')
        print('训练集图片共', str(train_data.shape[0]), '张')
        vali_data = My_CNN.load_data(self.dir, 'vali_data')
        print('验证集已读入！')
        print('验证集图片共', str(vali_data.shape[0]), '张')
        print('正在打乱数据集顺序...')
        train_index = np.arange(train_data.shape[0])
        vali_index = np.arange(vali_data.shape[0])
        np.random.shuffle(train_index)
        np.random.shuffle(vali_index)
        train_data = train_data[train_index, :, :, :]
        vali_data = vali_data[vali_index, :, :, :]
        train_label = self.train_label[train_index, :]
        vali_label = self.test_label[vali_index, :]


        print('正在训练网络...')
        print('epoch: ', str(epoch))
        print('batch_size: ', str(batch))
        if auto_save:
            print('自动保存 开启')
        else:
            print('自动保存 关闭')
        if early_save:
            print('提前终止 开启')
        else:
            print('提前终止 关闭')

        start_time = time.time()

        if early_save:
            # 提前终止，以验证集损失为标准
            history = self.model.fit(train_data, train_label, batch_size=batch, epochs=epoch,
                        verbose=2, validation_data=(vali_data, vali_label),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, restore_best_weights=True)])
        else:
            history = self.model.fit(train_data, train_label, batch_size=batch, epochs=epoch,
                                     verbose=2, validation_data=(vali_data, vali_label))

        end_time = time.time()
        time_cost = end_time-start_time
        history.history.update({'time': time_cost})

        # 保存在历史记录中的数据：训练正确率、训练误差、验证正确率、验证误差、训练时间

        if auto_save:
            self.save_data(history.history, hist_path, 'history')
            self.model.save(model_path)

    def conf_matrix(self):
        '''
        计算混淆矩阵
        :return: 混淆矩阵
        '''
        pre_index = []
        ac_index = []
        conf_mat = self.model.predict(self.test_data)
        for each in conf_mat:
            pre_index.append(My_CNN.find_max(each) + 1)
        for each in self.test_label:
            ac_index.append(My_CNN.find_max(each) + 1)
        matrix = np.zeros((self.class_num, self.class_num))
        for i in range(len(pre_index)):
            matrix[pre_index[i] - 1, ac_index[i] - 1] += 1
        return matrix

    def run_model(self, model_path):
        '''
        将测试集输入到训练好的网络模型中，并保存混淆矩阵和正确率
        :param model_path:模型保存的路径
        :return:
        '''
        print('正在读入数据集...')
        test_data = My_CNN.load_data(self.dir, 'vali_data')
        print('测试集已读入！')
        print('测试集图片共', str(test_data.shape[0]), '张')

        print('正在打乱数据集顺序...')
        test_index = np.arange(test_data.shape[0])
        np.random.shuffle(test_index)
        test_data = test_data[test_index, :, :, :]
        test_label = self.test_label[test_index, :]
        self.test_data = test_data
        self.test_label = test_label
        print('当前模型: ', model_path)
        print('正在载入网络...')
        self.model = load_model(model_path, custom_objects={'LBC': LBC})
        loss, accuracy = self.model.evaluate(test_data, test_label, verbose=0)
        print('测试正确率: ', str(accuracy))
        print('混淆矩阵: ')
        conf_mat = My_CNN.conf_matrix(self)
        print(conf_mat)
