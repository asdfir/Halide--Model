import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lars
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"C:\.....")
data.drop(["Formula","Remark"],inplace=True,axis=1)
# data["a"] =data["a"].fillna(data["a"].mean())
# data["c"] =data["c"].fillna(data["c"].mean())
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=320)
for i in[x_train,x_test,y_train,y_test]:
    i.index = range(i.shape[0])
#模型
lars = Lars(n_nonzero_coefs=5, fit_intercept=False, normalize=False).fit(x_train,y_train)
petrain = lars.predict(x_train)
petest = lars.predict(x_test)
r2_test = r2_score(y_test,petest)
r2_train = r2_score(y_train,petrain)
print(r2_train,r2_test)