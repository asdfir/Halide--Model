import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve #画学习曲线的类
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","roce"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=320)
for i in[x_train,x_test,y_train,y_test]:
    i.index = range(i.shape[0])
print(y_test)

#随机森林
rf = RandomForestRegressor(n_estimators= 200,max_depth=4,random_state=30,min_samples_split =4).fit(x_train,y_train)
pred_tr = rf.predict(x_train)
pred_te = rf.predict(x_test)
r2_tr = r2_score(y_train,pred_tr)
r2_te = r2_score(y_test,pred_te)
mse = mean_squared_error(y_test,pred_te)
rmse = np.sqrt(mse)
   # score_tr.append(r2_tr)
   # score_te.append(r2_te)
# print("测试集最大分数：",max(score_te),"此时的n_:",score_te.index(max(score_te)))
# print("训练集最大分数：",max(score_tr),"此时的n_:",score_tr.index(max(score_tr)))
# plt.figure()
# plt.plot(range(1,201),score_tr,c = "red",label = "train")
# plt.plot(range(1,201),score_te,c="orange",label = "test")
# plt.xlabel('n_estimators')
# plt.ylabel('Score')
# plt.title('n_estimators Learning curve')
# plt.legend()
# plt.show()
print("训练集分数",r2_tr,"测试集分数",r2_te,"MSE",mse,"RMSE:",rmse)

 #特征重要性
# features = list(x_test.columns)
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# num_features = len(importances)
# plt.figure()
# plt.title("RF Feature importances")
# plt.bar(range(num_features), importances[indices], color="g", align="center")
# plt.xticks(range(num_features), [features[i] for i in indices], rotation='60')
# plt.xlim([-1, num_features])
# plt.tight_layout()
# plt.show()
# for i in indices:
#     print ("{0} - {1:.3f}".format(features[i], importances[i]))

#RFE递归消除
# model = RandomForestRegressor(n_estimators= 5,max_depth= 6,random_state = 30)
#
# feature_counts = []
# scores = []
# scores2 = []
# for i in range(x.shape[1],0,-1):
#     rfe = RFE(model, n_features_to_select=i, step=1).fit(x_train,y_train)
#     feature_counts.append(i)
#     pre_train = rfe.predict(x_train)
#     pre_test = rfe.predict(x_test)
#     score = r2_score(y_test,pre_test)
#     score_train = r2_score(y_train,pre_train)
#     scores.append(score)
#     scores2.append(score_train)
# feature_counts.reverse()
# scores.reverse()
# scores2.reverse()
# plt.figure(figsize=(10,6))
# plt.plot(feature_counts,scores,marker = 'o')
# plt.plot(feature_counts,scores2,marker = 'o')
# plt.text(feature_counts[4], scores[4], str(scores[4]), ha='center', va='bottom')
# plt.text(feature_counts[4],scores2[4],str(scores2[4]),ha = "center",va = "bottom")
# plt.xlabel("feature counts")
# plt.ylabel("score")
# plt.show()

# feature_names =x.columns.tolist()
# feature_ranking = rfe.ranking_

# ranked_features = [(feature_names[i], feature_ranking[i]) for i in range(len(feature_names))]
#
# ranked_features.sort(key=lambda x: x)
# print("Feature Ranking and Names:")
# for feature, ranking in ranked_features:
#     print(f"{feature}: {ranking}")


# selected_features = list(x.columns[rfe.support_])
# print(selected_features)
#RFE可视化
# feature_ranking = rfe.ranking_
# rfe_features = pd.DataFrame({'Feature':x_train.columns,"Ranking": feature_ranking}).sort_values(by="Ranking")
# select_features = []
# for i in range(len(rfe_features)):
#     current_feature = rfe_features.iloc[i]["Feature"]
#     select_features.append(current_feature)
#     x_train_subset = x_train[select_features]
#     x_test_subset = x_test[select_features]
#     rf_clf = RandomForestRegressor().fit(x_train_subset,y)
#     importance = rf_clf.feature_importances_[len(select_features) - 1]
#     y_pred_proda = rf_clf.predict(x_test_subset)
#     roc_score = r2_score(y_test,y_pred_proda)
#     select_features.loc[len(select_features)]=[
#         current_feature,
#         importance,
#         roc_score]
# print(select_features)
# rfe1 = rfe.support_
# re_pre= rfe.predict(x_test)
# re_pre2 = rfe.predict(x_train)
# re_te = r2_score(y_test,re_pre)

# re_tr = r2_score(y_train,re_pre2)
# print(rfe.n_features_)

#模型效果拟合图
# plt.figure()
# plt.scatter(y_test,pred_te,label = "test")
# plt.scatter(y_train,pred_tr,label = "train")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],'k--', lw=2)
# plt.legend()
# plt.xlabel('true value')
# plt.ylabel('predicted value')
# plt.title("Model Fit Effect")
 # plt.text(0,2.7,"train_R2 = 0.85",fontsize = 10)
# plt.text(0,2.5,"test_R2 = 0.37",fontsize= 10)
# plt.grid()
# plt.show()
#特征重要性
# importance = rf.feature_importances_
# features = x_train.columns
# indices = np.argsort(importance)
# plt.figure(figsize=(10, 6))
# plt.title('RF Feature Importance')
# plt.barh(range(len(indices)), importance[indices], color='#1f77b4', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.tight_layout()
# plt.show()

#拟合学习曲线
# def plot_learning_curve(estimator,#模型评估器
#                         title,#图的标题
#                         x,y,
#                         ax = None# 选择子图
#                        ,ylim = None#纵坐标范围
#                        , cv = None
#                        ,n_jobs = None):#计算资源
#     train_size,train_score,test_score = learning_curve(estimator,x,y,cv = cv,n_jobs = n_jobs)
#     if ax == None:
#         ax = plt.gca()
#     else:
#         ax = plt.figure()
#     ax.set_title(title)#设置标题
#     if ylim is not None:
#         ax.set_ylim(*ylim)
#     ax.set_xlabel("Training examples")#横坐标名字
#     ax.set_ylabel("Score")#纵坐标名字
#     ax.grid()
#     ax.plot(train_size,np.mean(train_score,axis = 1),"o-",c = "g",label = "Train score")
#     ax.plot(train_size,np.mean(test_score,axis = 1),"o-",c = "r",label = "Test score")
#     ax.legend(loc = "best")
#     return ax
# cv = ShuffleSplit(n_splits = 50,#把数据分成多少分
#                   test_size = 0.2,#0.2*50作为交叉验证的测试集
#                   random_state = 0)
# plot_learning_curve(rf,"XGB",x,y,cv = cv,ax = None)
# plt.show()

