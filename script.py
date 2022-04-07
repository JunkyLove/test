import warnings

warnings.filterwarnings("ignore")
from create_samples import *
from FOA import *

from Adaboost import *

p1=r'.\t-3.xlsx'
p2=r'.\t-3_normal.xlsx'
p=[]
p.append(p1)
p.append(p2)
features=[]
targets=[]
number_of_original_samples=13

features_train, targets_train, features_test,targets_test=create_samples(p,smote=True,smote_index=0,number_of_not_to_smote=number_of_original_samples)
clf=SVM.SVM(1, 20, features_train, targets_train,features_test, targets_test)
yy,BestC,BestGamma=FOA(5,50 , clf)   #Best: 10,100 if u r using t-2 data

print('经FOA优化后所得分类器的混淆矩阵为：\n',clf.report())
tmp0=clf.Predict_results
clf.Predict_results=clf.clf.predict(clf.Features_train)
tmp1=clf.Targets_test
clf.Targets_test=clf.Targets_train
print('若以训练集测试，则混淆矩阵为：\n',clf.report())
clf.Predict_results=tmp0
clf.Targets_test=tmp1

print('开始Adaboost优化')

my_adaboost_clf(clf.Targets_train, clf.Features_train, clf.Targets_test, clf.Features_test, M=20, Adaclf=clf)


