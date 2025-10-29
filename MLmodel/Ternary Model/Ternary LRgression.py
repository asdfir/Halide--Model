import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import RFE


data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","a","c","MN_Mis"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=50)
for i in[x_train,x_test,y_train,y_test]:
    i.index = range(i.shape[0])

clf = LinearRegression().fit(x_train,y_train)
pred_tr = clf.predict(x_train)
pred_te = clf.predict(x_test)
r2_tr =r2_score(y_train,pred_tr)
r2_te = r2_score(y_test,pred_te)
print(r2_tr,r2_te)




