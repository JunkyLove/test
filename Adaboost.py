import numpy as np
import SVM
from sklearn import tree

def my_adaboost_clf(Y_train, X_train, Y_test, X_test, M=20, Adaclf=None):

    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        print('在Adaboost算法中的分类器混淆矩阵为：\n', Adaclf.report())
        Adaclf.clf.fit(X_train, Y_train,sample_weight=w)
        Adaclf.Predict_results=Adaclf.clf.predict(X_test)
        print('在Adaboost算法中的分类器混淆矩阵为：\n',Adaclf.report())

        pred_train_i = Adaclf.clf.predict(X_train)
        pred_test_i = Adaclf.clf.predict(X_test)

        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        print("weak_clf_%02d train acc: %.4f"
         % (i + 1, 1 - sum(miss) / n_train))

        # Error
        err_m = np.dot(w, miss)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        miss2 = [x if x==1 else -1 for x in miss] # -1 * y_i * G(x_i): 1 / -1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)

        # Add to prediction
        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)

    pred_train = (pred_train > 0) * 1
    pred_test = (pred_test > 0) * 1

    print("My AdaBoost clf train accuracy: %.4f" % (sum(pred_train == Y_train) / n_train))
    print("My AdaBoost clf test accuracy: %.4f" % (sum(pred_test == Y_test) / n_test))
    Adaclf.Predict_results=pred_test
    print('当前Ada分类器混淆矩阵：\n', Adaclf.report())
