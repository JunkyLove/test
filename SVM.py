from sklearn import svm
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import SMOTE
import numpy as np

class SVM:
    def __init__(self,C, gamma, features_train, targets_train, features_test, targets_test):
        self.C=C
        self.Gamma=gamma
        #self.Features=features
        #self.Targets=targets

        self.Predict_results=None

        self.Features_train=features_train
        self.Features_test=features_test
        self.Targets_train=targets_train
        self.Targets_test=targets_test

        self.clf=None


        # 标准化sca转换
        scaler = StandardScaler()

        # 训练标准化对象
        scaler.fit(self.Features_train)
        scaler.fit(self.Features_test)
        # 转换数据集，一般是归一化处理
        self.Features_train = scaler.transform(self.Features_train)
        self.Features_test=scaler.transform(self.Features_test)

        self.clf = svm.SVC(self.C, kernel='rbf', gamma=self.Gamma, decision_function_shape='ovr')

        # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
        # 将训练数据放到里面，到这一步的时候 svm已经完成了
        self.clf.fit(self.Features_train, self.Targets_train)

        # 这边是开始预测了，输入的测试集
        self.Predict_results = self.clf.predict(self.Features_test)
        #print(classification_report(self.Targets_test, self.Predict_results))
        #print('wait')
        # 预测结果和原来结果进行的一个对比

        #print(accuracy_score(predict_results, target_test))


    def recalculate_recall(self):
        self.clf=svm.SVC(C=self.C,kernel='rbf',gamma=self.Gamma,decision_function_shape='ovr')
        self.clf.fit(self.Features_train,self.Targets_train)
        self.Predict_results = self.clf.predict(self.Features_test)
        report=classification_report(self.Targets_test, self.Predict_results,digits=8)
        lines=report.split('\n')
        line=(lines[3].split(' '))
        recall_of_1=float(line[14])
        return recall_of_1
    def report(self):
        return classification_report(self.Targets_test,self.Predict_results,digits=8)
        # print(precision_score(predict_results,target_test,average=None))
        # 参数结果 包括准确率 召回率 f1
        # s=classification_report(target_test,predict_results)
        # print(classification_report(target_test, predict_results))
