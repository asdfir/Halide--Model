import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","a","c","MN_Mis"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=60)
#KRR模型
krr = KernelRidge(
    alpha=0.4,          # 正则化强度
    kernel='rbf',       # 核函数类型
    gamma=0.1,          # RBF核参数
    kernel_params=None
).fit(x_train,y_train)
petrain = krr.predict(x_train)
petest = krr.predict(x_test)
r2_test = r2_score(y_test,petest)
r2_train = r2_score(y_train,petrain)
print(r2_train,r2_test)
