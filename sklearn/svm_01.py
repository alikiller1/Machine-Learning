#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 22:51
# @Author  : liuqh
# @Software: PyCharm
#导入svm和数据集
from sklearn import svm,datasets
import numpy as np

#调用SVC()
clf = svm.SVC()
#载入鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
#fit()训练
clf.fit(X,y)
#predict()预测
pre_y = clf.predict(X[5:10])
print(pre_y)
print(y[5:10])
#导入numpy

a=[[5.1,2.9,1.8,3.6]];
a=[[5.1,2.9,1.8,3.6]];
test = np.array([[5.1,2.9,1.8,3.6]])
#对test进行预测
test_y = clf.predict(test)
print(test_y)

x1=[[1,1,1],[0,1,1],[0,1,0],
   [1,0,0],[2,2,2],[3,2,3]]
y1=[[1,0,0,
    1,1,1]]
x1=np.mat(x1)
y1=np.mat(y1).T
clf.fit(np.mat(x1),np.mat(y1))

test2=np.array([[4,0,4],[4,0,2],[4,3,4]])

print(clf.predict(test2))