import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFE

data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","roce"],inplace=True,axis=1)
# data["a"] =data["a"].fillna(data["a"].mean())
# data["c"] =data["c"].fillna(data["c"].mean())
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=320)
for i in[x_train,x_test,y_train,y_test]:
    i.index = range(i.shape[0])

# score_tr= []
# score_te=[]
# C_range = np.linspace(0.01,30,50)
# for i in C_range:
clf = SVR(kernel="rbf",cache_size=5000,gamma= 0.0042919,C =1.234).fit(x_train,y_train)
pred_tr =clf.predict(x_train)
pred_te =clf.predict(x_test)
r2_tr = r2_score(y_train,pred_tr)
r2_te = r2_score(y_test,pred_te)
print("训练集分数 %f，测试集分数 %f" % (r2_tr,r2_te))
#   score_tr.append(r2_tr)
#   score_te.append(r2_te)
# print("测试集最大分数：",max(score_te),"此时的gamma:",C_range[score_te.index(max(score_te))])
# print("训练集最大分数：",max(score_tr),"此时的gamma:",C_range[score_tr.index(max(score_tr))])
# plt.figure()
# plt.plot(C_range,score_tr,c = "red",label = "train")
# plt.plot(C_range,score_te,c="orange",label = "test")
# plt.title("C learn curve")
# plt.xlabel("gamma")
# plt.ylabel("Score")
# plt.legend()
# plt.show()









