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
data.drop(["Formula","Remark","roce"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=320)
for i in[x_train,x_test,y_train,y_test]:
    i.index = range(i.shape[0])

clf = LinearRegression().fit(x_train,y_train)
pred_tr = clf.predict(x_train)
pred_te = clf.predict(x_test)
r2_tr =r2_score(y_train,pred_tr)
r2_te = r2_score(y_test,pred_te)
print(r2_tr,r2_te)
#特征拟合程度
# plt.figure()
# plt.scatter(y_test,pred_te,label = "test")
# plt.scatter(y_train,pred_tr,label = "train")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],'k--', lw=2)
# plt.legend()
# plt.xlabel('true value')
# plt.ylabel('predicted value')
# plt.title("Model Fit Effect")
# plt.grid()
# plt.show()
#预测值与真实值的对比
# plt.plot(range(len(y_test)),sorted(y_test),c="black",label = "data")
# plt.plot(range(len(pred_tr)),sorted(pred_tr),c="red",label = "predict")
# plt.legend()
# plt.show()

#



