#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 22:51
# @Author  : liuqh
# @Software: PyCharm
#导入svm和数据集
from sklearn import svm,datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy import float64
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer


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


clf2 = svm.SVC()

x1=[[-1,1,1],[2,3,1],[4,1,0],
   [8,0,0],[1,1,2],[3,5,3],
    [-8,10,0],[3,3,5],[8,9,1],
    [3, 3, 5], [4, 3, 0], [0, 9, 1],[0,0,0]]
y1=[[0,1,0,
    0,1,1,
    0,1,1,
    1,0,0,0]]
x1=np.mat(x1)
y1=np.mat(y1)

binarizer = preprocessing.Binarizer().fit(x1)
x1_binarizer=binarizer.transform(x1);


clf2.fit(x1_binarizer,y1.T)
#clf2.fit(x1,y1.T)
test2=np.mat([[1,1,1],[0,0,0],[3,3,3],[1,3,5],[0,2,3]])

test2_binarizer=binarizer.transform(test2);

print("{{{{{{{{{{{{{{{{{{{")
print(clf2.predict(test2_binarizer))
#print(clf2.predict(test2))

print('==============================')
print(x1.max(axis=0))
print(x1.min(axis=0))



#MinMaxScaler
X_std = (x1 - x1.min(axis=0)) / (x1.max(axis=0) - x1.min(axis=0))
print('--------------')
print(X_std);

min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x1)
print('--------------')
print(x_minmax)

##MaxAbsScaler
max_abs_scaler = preprocessing.MaxAbsScaler()
x_train_maxsbs = max_abs_scaler.fit_transform(x1)
print('--------------')
print(x_train_maxsbs)


##正则化Normalization
x2=[[ 1,-1,2],[ 2,0,0],[0,1,-1]]
normalizer = preprocessing.Normalizer().fit(x2)
x_normalized=normalizer.transform(x2)
print('+++++++++++++++++++')
print (x_normalized)

##二值化–特征的二值化
x3 = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

binarizer = preprocessing.Binarizer().fit(x3)
x_binarizer= binarizer.transform(x3)
print('+++++++++++++++++++')
print(x_binarizer)

print(np.log1p(2))


testdata = pd.DataFrame({'pet': ['cat', 'dog', 'dog', 'fish'],'age': [4 , 6, 3, 3],'salary':[4, 1, 1, 1]})
a1 = OneHotEncoder(sparse = False).fit_transform( testdata[['age']] )
a2 = OneHotEncoder(sparse = False).fit_transform( testdata[['salary']])
print("----------------------------+++")
print(testdata)
final_output = np.hstack((a1,a2))
print("----------------------------+++")
print(final_output)

print("----------------------------+++")
a = LabelEncoder().fit_transform(testdata['pet'])
print(a);
print(a.reshape(-1,1).shape)
OneHotEncoder(sparse=False).fit_transform(a.reshape(-1, 1))  # 注意: 这里把 a 用 reshape 转换成 2-D array

# 方法二: 直接用 LabelBinarizer()

a3=LabelBinarizer().fit_transform(testdata['pet'])
print(a3)

a4=pd.get_dummies(testdata,columns=testdata.columns)

print(a4)