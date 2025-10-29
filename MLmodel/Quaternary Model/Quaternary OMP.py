import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","roce"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=320)

#OMP模型
omp = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5,
    tol=None,
    fit_intercept=True
).fit(x_train,y_train)
petrain = omp.predict(x_train)
petest = omp.predict(x_test)
r2_test = r2_score(y_test,petest)
r2_train = r2_score(y_train,petrain)
print(r2_test,r2_train)