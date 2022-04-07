from libsvm.svm import *
from libsvm.svmutil import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from SVM import *
import numpy as np



def Fitness(C,gamma,clf):
    clf.C=C
    clf.Gamma=gamma
    #print(clf.recall())
    return clf.recalculate_recall()

def FOA(maxgen,sizepop,clf):       #C，gamma初始值为 1， 20
    # 初始果蝇位置
    C_axis = clf.C
    Gamma_axis = clf.Gamma

    # 果蝇寻优开始，利用嗅觉寻找食物
    X = []
    Y = []
    #D = []
    #S = []
    print('初始召回率：',Fitness(clf.C,clf.Gamma,clf))
    print('初始混淆矩阵：\n',clf.report())
    Smell = []
    for i in range(sizepop):
        # 赋予果蝇个体利用嗅觉搜寻食物之随机方向与距离
        potential_X=C_axis + 2 * np.random.rand() - 1
        potential_Y = Gamma_axis + 20 * np.random.rand() - 10
        X.append(potential_X if potential_X>0 else (-1)*potential_X )
        Y.append(potential_Y if potential_Y>0 else (-1)*potential_Y)

        # 由于无法得知食物位置，因此先估计与原点的距离（Dist），再计算味道浓度判定值（S），此值为距离的倒数
        #D.append((X[i]**2 + Y[i]**2)**0.5)
        #S.append(1 / D[i])

        # 味道浓度判定值（S）代入味道浓度判定函数（或称为Fitness function），以求出该果蝇个体位置的味道浓度（Smell(i))
        Smell.append(Fitness(X[i],Y[i],clf))

    # 找出此果蝇群里中味道浓度最大的果蝇（求极大值）
    bestSmell, bestindex = max(Smell),Smell.index(max(Smell))
    print('首轮优化后召回率：',bestSmell)

    # 保留最佳味道浓度值与x，y的坐标，此时果蝇群里利用视觉往该位置飞去
    X_axis = X[bestindex]
    Y_axis = Y[bestindex]
    Smellbest = bestSmell

    # 果蝇迭代寻优开始
    yy = []
    Xbest = []
    Ybest = []
    for g in range(maxgen):
        # 赋予果蝇个体利用嗅觉搜寻食物的随机方向和距离
        for i in range(sizepop):
            # 赋予果蝇个体利用嗅觉搜寻食物之随机方向与距离
            potential_X = X_axis + 2 * np.random.rand() - 1
            potential_Y=Y_axis + 20 * np.random.rand() - 10
            X[i] = potential_X if potential_X>0 else (-1)*potential_X
            Y[i] = potential_Y if potential_Y>0 else (-1)*potential_Y

            # 由于无法得知食物位置，因此先估计与原点的距离（Dist），再计算味道浓度判定值（S），此值为距离的倒数
            #D[i] = (X[i]**2 + Y[i]**2)**0.5
            #S[i] = 1 / D[i]

            # 味道浓度判定值（S）代入味道浓度判定函数（或称为Fitness function），以求出该果蝇个体位置的味道浓度（Smell(i))
            Smell[i] =Fitness(X[i],Y[i],clf)

        # 找出此果蝇群里中味道浓度最大的果蝇（求极大值）
        bestSmell, bestindex = max(Smell),Smell.index(max(Smell))

        # 判断味道浓度是否优于前一次迭代味道浓度，若是则保留最佳味道浓度值与x，y的坐标，此时果蝇群体利用视觉往该位置飞去
        if bestSmell > Smellbest:
            print('当前最佳召回率：',bestSmell)
            X_axis = X[bestindex]
            Y_axis = Y[bestindex]
            Smellbest = bestSmell

        # 每次最优Semll值记录到yy数组中，并记录最优迭代坐标
        yy.append(Smellbest)
        Xbest.append(X_axis)
        Ybest.append(Y_axis)
    clf.C=Xbest[yy.index(max(yy))]
    clf.Gamma = Ybest[yy.index(max(yy))]
    #重置clf
    clf.clf = svm.SVC(C=clf.C, kernel='rbf', gamma=clf.Gamma, decision_function_shape='ovr')
    clf.clf.fit(clf.Features_train, clf.Targets_train)
    clf.Predict_results=clf.clf.predict(clf.Features_test)

    return yy, Xbest, Ybest



'''
maxgen = 200
sizepop = 50
yy, Xbest, Ybest = FOA(maxgen,sizepop)
ax1 = plt.subplot(121)
ax1.plot(yy)
ax1.set(xlabel = 'Iteration Number',ylabel = 'Smell',title = 'Optimization process')
ax2 = plt.subplot(122)
ax2.plot(Xbest,Ybest)
ax2.set(xlabel = 'X-axis',ylabel = 'Y-axis',title = 'Fruit fly flying route')
plt.show()
'''
