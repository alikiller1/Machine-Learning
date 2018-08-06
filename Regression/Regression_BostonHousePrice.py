# coding: utf-8

# # 波士顿房价预测案例
# 
# 在这个案例中，我们将利用波士顿郊区的房屋信息数据训练和测试一个模型，并对模型的性能和预测能力进行测试。
# 
# 该数据集来自UCI机器学习知识库。波士顿房屋这些数据于1978年开始统计，共506个数据点，涵盖了麻省波士顿不同郊区房屋13种特征和房价的信息。
# 
# 本项目将原始数据集存为csv格式，方便调用pandas做数据分析。

# ## 1、导入必要的工具包

# In[1]:


import numpy as np  # 矩阵操作
import pandas as pd # SQL数据处理

from sklearn.metrics import r2_score  #评价回归预测模型的性能

import matplotlib.pyplot as plt   #画图
import seaborn as sns




# ## 2、数据探索

# ### 2.1 读取数据

# In[2]:


# path to where the data lies
dpath = './data/'
data = pd.read_csv(dpath +"boston_housing.csv")

#通过观察前5行，了解数据每列（特征）的概况
data.head()


# ###  2.2 数据基本信息
# 样本数目、特征维数
# 每个特征的类型、空值样本的数目、数据类型

# In[3]:


data.shape


# In[4]:


## 各属性的统计信息（样本数目、均值、标准差、最小值、最大值、1/4分位数、中值（1/2分位数）、3/4分位数）
# 只计算数值型特征的统计信息（int、float）

data.describe()


# ### 2.3 数据探索
# 请见另一个文件：FE_BostonHousePrice.pynb
# 
# 对数据的探索有助于我们在第三步中根据数据的特点选择合适的模型类型

# ### 2.4 数据准备

# In[5]:


# 从原始数据中分离输入特征x和输出y
y = data['MEDV'].values
print(y.shape)
X = data.drop(['MEDV','INDUS','AGE'],axis = 1)
print(X.shape)


# 当数据量比较大时，可用train_test_split从训练集中分出一部分做校验集；
# 样本数目较少时，建议用交叉验证
# 在线性回归中，留一交叉验证有简便计算方式，无需显式交叉验证
# 
# 下面将训练数据分割成训练集和测试集，只是让大家对模型的训练误差、校验集上的测试误差估计、和测试集上的测试误差做个比较，实际任务中无需这么处理。

# In[6]:


#将数据分割训练数据与测试数据
from sklearn.model_selection import train_test_split

# 随机采样25%的数据构建测试样本，其余作为训练样本
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.2)


# ### 2.5 数据预处理／特征工程
# 
# 特征工程是实际任务中特别重要的环节。
# 
# scikit learn中提供的数据预处理功能：
# http://scikit-learn.org/stable/modules/preprocessing.html
# http://scikit-learn.org/stable/modules/classes.html#module- sklearn.feature_extraction

# In[7]:


#发现各特征差异较大，需要进行数据标准化预处理
#标准化的目的在于避免原始特征值差异过大，导致训练得到的参数权重不归一，无法比较各特征的重要性


# In[8]:


# 数据标准化
from sklearn.preprocessing import StandardScaler

# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

#y_train = ss_y.fit_transform(y_train)
#y_test = ss_y.transform(y_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))


# ## 3、确定模型类型

# ### 3.1 尝试缺省参数的线性回归

# In[9]:


# 线性回归
#class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化
lr = LinearRegression()

# 训练模型参数
lr.fit(X_train, y_train)

# 预测，下面计算score会自动调用predict
lr_y_predict = lr.predict(X_test)
lr_y_predict_train = lr.predict(X_train)

#显示特征的回归系数
lr.coef_


# #### 3.1.1 模型评价

# In[10]:


# 使用LinearRegression模型自带的评估模块（r2_score），并输出评估结果

#测试集
print ('The value of default measurement of LinearRegression on test is', lr.score(X_test, y_test))

#训练集
print ('The value of default measurement of LinearRegression on train is', lr.score(X_train, y_train))


# In[11]:


#在训练集上观察预测残差的分布，看是否符合模型假设：噪声为0均值的高斯噪声
f, ax = plt.subplots(figsize=(7, 5)) 
f.tight_layout() 
ax.hist(y_train - lr_y_predict_train,bins=40, label='Residuals Linear', color='b', alpha=.5); 
ax.set_title("Histogram of Residuals") 
ax.legend(loc='best');


# 残差分布和高斯分布比较匹配，但还是左skew，可能是由于数据集中有16个数据的y值为最大值，有噪声（预测残差超过2.5）

# In[12]:


#还可以观察预测值与真值的散点图
plt.figure(figsize=(4, 4))
plt.scatter(y_train, lr_y_predict_train)
plt.plot([-3, 3], [-3, 3], '--k')   #数据已经标准化，3倍标准差即可
plt.axis('tight')
plt.xlabel('True price')
plt.ylabel('Predicted price')
plt.tight_layout()


# In[13]:


# 线性模型，随机梯度下降优化模型参数
# 随机梯度下降一般在大数据集上应用，其实本项目不适合用
from sklearn.linear_model import SGDRegressor

# 使用默认配置初始化线
sgdr = SGDRegressor(max_iter=1000)

# 训练：参数估计
sgdr.fit(X_train, y_train)

# 预测
#sgdr_y_predict = sgdr.predict(X_test)

sgdr.coef_


# In[14]:


# 使用SGDRegressor模型自带的评估模块，并输出评估结果
print ('The value of default measurement of SGDRegressor on test is', sgdr.score(X_test, y_test))
print ('The value of default measurement of SGDRegressor on train is', sgdr.score(X_train, y_train))


# In[15]:


#这里由于样本数不多，SGDRegressor可能不如LinearRegression。 sklearn建议样本数超过10万采用SGDRegressor


# ### 3.2 正则化的线性回归（L2正则 --> 岭回归）

# In[16]:


#岭回归／L2正则
#class sklearn.linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, 
#                                  normalize=False, scoring=None, cv=None, gcv_mode=None, 
#                                  store_cv_values=False)
from sklearn.linear_model import  RidgeCV

alphas = [0.01, 0.1, 1, 10,20, 40, 80,100]
reg = RidgeCV(alphas=alphas, store_cv_values=True)   
reg.fit(X_train, y_train)       


# In[17]:

print('reg.cv_values_ type=',type(reg.cv_values_));
cv_values=reg.cv_values_;
mse_mean = np.mean(reg.cv_values_, axis = 0)
print(mse_mean)
plt.plot(np.log10(alphas), mse_mean.reshape(len(alphas),1)) 
plt.plot(np.log10(reg.alpha_)*np.ones(3), [0.28, 0.29, 0.30])
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print ('alpha is:', reg.alpha_)
reg.coef_


# In[18]:


# 使用LinearRegression模型自带的评估模块（r2_score），并输出评估结果
print ('The value of default measurement of RidgeRegression is', reg.score(X_test, y_test))


# ### 3.3 正则化的线性回归（L1正则 --> Lasso）

# In[19]:


#### Lasso／L1正则
# class sklearn.linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, 
#                                    normalize=False, precompute=’auto’, max_iter=1000, 
#                                    tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=1,
#                                    positive=False, random_state=None, selection=’cyclic’)
from sklearn.linear_model import LassoCV

alphas = [0.01, 0.1, 1, 10,100]

lasso = LassoCV(alphas=alphas)   
lasso.fit(X_train, y_train)       


# In[20]:


mses = np.mean(lasso.mse_path_, axis = 1)
plt.plot(np.log10(lasso.alphas_), mses) 
#plt.plot(np.log10(lasso.alphas_)*np.ones(3), [0.3, 0.4, 1.0])
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()    
            
print ('alpha is:', lasso.alpha_)
lasso.coef_  


# In[21]:


#在本任务中，最佳alpha为参数grid的最左端，最好再继续检查比当前更小的alpha是否会更好


# In[22]:


# 使用LinearRegression模型自带的评估模块（r2_score），并输出评估结果
print ('The value of default measurement of Lasso Regression on test is', lasso.score(X_test, y_test))
print ('The value of default measurement of Lasso Regression on train is', lasso.score(X_train, y_train))

