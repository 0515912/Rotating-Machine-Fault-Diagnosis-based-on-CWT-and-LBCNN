# Machine Learning methods
# author:  Matthew Lin
# date: 10/03/2019
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from ELM import HiddenLayer
from sklearn import metrics

class ML_Trainer():
    '''
    使用不同的机器学习算法训练
    '''

    @staticmethod
    def load_data(dir, name):
        '''
        载入数据集
        :param dir:保存路径
        :param name: 保存名称
        :return: 数据集
        '''
        file = open(dir + name + u'.txt', 'rb+')
        return pickle.load(file)

    @staticmethod
    def generate_label(class_num, class_vol):
        num = class_num * 1
        train_vol = int(class_vol * 0.8)
        test_vol = int(class_vol * 0.1)
        print('正在生成训练集标签...')
        train_gross = num * train_vol
        train_label = [int(np.ceil((x + 1) / train_vol) - 1) for x in range(train_gross)]
        print('正在生成测试集标签...')
        test_gross = num * test_vol
        test_label = [int(np.ceil((x + 1) / test_vol) - 1) for x in range(test_gross)]
        return train_label, test_label


    def __init__(self, pic_size, channel, class_num, class_vol, dir):
        # 载入数据
        print('正在加载数据...')
        print('图片尺寸: ', [pic_size, pic_size, channel])
        print('分类个数: ', class_num)
        print('每类样本: ', class_vol)
        print('当前路径: ', dir)
        train_label, test_label = ML_Trainer.generate_label(class_num, class_vol)
        train_data = ML_Trainer.load_data(dir, 'train_data')
        test_data = ML_Trainer.load_data(dir, 'test_data')
        train_data = np.reshape(train_data, [-1, pic_size * pic_size * channel])
        test_data = np.reshape(test_data, [-1, pic_size * pic_size * channel])
        self.train_label = train_label
        self.train_data = train_data
        self.test_label = test_label
        self.test_data = test_data


    def SVM_trainer(self):
        '''
        SVM分类器
        :return:
        '''
        model = SVC(kernel='linear')
        print('正在训练SVM模型...')
        model.fit(self.train_data, self.train_label)
        print('正在测试SVM模型...')
        result = model.score(self.test_data, self.test_label)
        print('SVM结果: ', result)

    def PCA_SVM(self, components=1600):
         '''
         PCA-SVM分类器
         :param self:
         :param components: PCA所取主成分个数
         :return:
         '''
         print('正在进行PCA处理...')
         model = PCA(n_components=components)
         x_train = model.fit_transform(self.train_data)
         x_test = model.fit_transform(self.test_data)
         model = SVC(kernel='linear')
         print('正在训练PCA-SVM模型...')
         model.fit(x_train, self.train_label)
         print('正在测试PCA-SVM模型...')
         result = model.score(x_test, self.test_label)
         print('PCA-SVM结果: ', result)

    def KNN_trainer(self):
         '''
         K近邻分类器
         :param self:
         :return:
         '''
         model = KNeighborsClassifier()
         print('正在训练KNN模型...')
         model.fit(self.train_data, self.train_label)
         print('正在测试KNN模型...')
         result = model.score(self.test_data, self.test_label)
         print('KNN结果: ', result)

    def AdaBoost_trainer(self, estimator=200):
        '''
        Adaboost分类器
        :param estimator: 子分类器数量
        :return:
        '''
        model = AdaBoostClassifier(n_estimators=estimator)
        print('正在训练AdaBoost模型...')
        model.fit(self.train_data, self.train_label)
        print('正在测试AdaBoost模型...')
        result = model.score(self.test_data, self.test_label)
        print('Adaboost结果: ', result)

    def ELM(self):
        print('正在训练ELM模型...')
        classifier = HiddenLayer(self.train_data, 4000)
        classifier.classifisor_train(self.train_label)
        print('正在测试ELM模型...')
        result = classifier.classifisor_test(self.test_data)
        print('ELM结果: ', metrics.accuracy_score(self.test_label, result))



if __name__ == '__main__':
    # MFPT数据集
    dir =  'F:\毕业设计\MFPT\时频图\数据记录\\'
    model = ML_Trainer(40, 3, 16, 1000, dir)
    # SVM
    model.SVM_trainer()
    # KNN
    model.KNN_trainer()
    # Adaboost
    model.AdaBoost_trainer()
    # ELM
    model.ELM()

