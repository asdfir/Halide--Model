import numpy as np
from sklearn.linear_model import ElasticNet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","roce"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)
for i in[x_train,x_test,y_train,y_test]:
    i.index = range(i.shape[0])

en = ElasticNet(alpha=0.7, l1_ratio=1).fit(x_train,y_train)
petrain = en.predict(x_train)
petest = en.predict(x_test)
r2_test = r2_score(y_test,petest)
r2_train = r2_score(y_train,petrain)
print(r2_train,r2_test)
