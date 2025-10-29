import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

data = pd.read_csv(r"C:\.....")
data.drop(["Formula","Remark"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=90)
#决策树模型
dt_reg = DecisionTreeRegressor(max_depth=12 ,random_state = 250,min_samples_split =9).fit(x_train,y_train)
petrain = dt_reg.predict(x_train)
petest = dt_reg.predict(x_test)
r2_test = r2_score(y_test,petest)
r2_train = r2_score(y_train,petrain)
print("训练集分数:",r2_train,"测试集分数：",r2_test)
#递归消除
# model = DecisionTreeRegressor(max_depth=7,random_state = 452,min_samples_split =9)
# feature_counts = []
# scores = []
# scores2 = []
# for i in range(x.shape[1],0,-1):
#     rfe = RFE(model, n_features_to_select=i, step=1).fit(x_train,y_train)
#     feature_counts.append(i)
#     pre_test = rfe.predict(x_test)
#     pre_train = rfe.predict(x_train)
#     score = r2_score(y_test,pre_test)
#     score_train = r2_score(y_train, pre_train)
#     scores2.append(score_train)
#     scores.append(score)
# feature_counts.reverse()
# scores.reverse()
# plt.figure(figsize=(10,6))
# plt.plot(feature_counts,scores,marker = 'o',lw = "4",label = "test set")
# plt.plot(feature_counts,scores2,marker = 'o',lw = "4",label = "train set")
# plt.title("RFE score")
# plt.xlabel("Number of features selected")
# plt.ylabel("score")
# # plt.text(feature_counts[4], scores[4], str(scores[4]), ha='center', va='bottom')
# # plt.text(feature_counts[4],scores2[4],str(scores2[4]),ha = "center",va = "bottom")
# plt.legend()
# # plt.savefig('RFE.png')
# plt.show()
# # 获取特征排名和对应特征
# feature_names =x.columns.tolist()
# feature_ranking = rfe.ranking_
# ranked_features = [(feature_names[i], feature_ranking[i]) for i in range(len(feature_names))]# 创建排名和特征的对应关系
# ranked_features.sort(key=lambda x: x)
# print("Feature Ranking and Names:")
# for feature, ranking in ranked_features:
#     print(f"{feature}: {ranking}")
#特征重要性图
# importance = dt_reg.feature_importances_
# features = x_train.columns
# indices = np.argsort(importance)
# plt.figure(figsize=(10, 6))
# plt.title('DT Feature Importance')
# plt.barh(range(len(indices)), importance[indices], color='#1f77b4', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.tight_layout()
# plt.show()