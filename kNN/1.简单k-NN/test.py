#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/8 10:36
# @Author  : liuqh
# @Software: PyCharm

import pandas as pd
import numpy as np



def f2():
    data = pd.read_excel("data.xls");
    data = data.set_index("id");


    data1=data.copy();
    data1['age']= data1['age'] + 1;

    data2 = pd.merge(data,data1,right_on='id',left_on='age')

    print(data2)



def f1():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]]);
    test = [101, 20];
    print(test - group)
    print((test - group) ** 2)
    dist = np.sum((test - group) ** 2, axis=1) ** 0.5
    print(dist)

def f3():
    a=np.NAN;
    b=np.nan;
    c=None;
    d='';
    print(pd.isna(a))
    print(pd.isna(b))
    print(pd.isna(c))
    print(pd.isna(d))
    if a:
        print('1')
    if b:
        print('2')
    if c:
        print('3')
    if d:
        print('4')
    if not a:
        print('11')
    if not b:
        print('22')
    if not c:
        print('33')
    if not d:
        print('44')

def f4():
    data = pd.read_excel("data.xls");
    print(data)
    cond1=data['age']==1;
    cond2=data['id']==1;
    data=data[~(cond1&cond2)]
    print(data)



if __name__ == '__main__':
    f4();


