import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","a","c","MN_Mis"],inplace=True,axis=1)
# data["a"] =data["a"].fillna(data["a"].mean())
# data["c"] =data["c"].fillna(data["c"].mean())
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)
#决策树模型
dt_reg = DecisionTreeRegressor(max_depth=5,random_state = 30,min_samples_split =12).fit(x_train,y_train)
petrain = dt_reg.predict(x_train)
petest = dt_reg.predict(x_test)
r2_test = r2_score(y_test,petest)
r2_train = r2_score(y_train,petrain)
mse = mean_squared_error(y_test,petest)
rmse = np.sqrt(mse)
print("训练集分数:",r2_train,"测试集分数：",r2_test,mse,rmse)
#递归消除
# model = DecisionTreeRegressor(max_depth=4,random_state = 30,min_samples_split =6)
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
# ranked_features.sort(key=lambda x: x)# 根据排名对特征进行排序
# print("Feature Ranking and Names:")# 打印排名和特征的对应关系
# for feature, ranking in ranked_features:
#     print(f"{feature}: {ranking}")
#网格搜索最优参数
# param_grid = {
#     'max_depth': [3, 4, 5, 6, 7, 8],
#     'min_samples_split': [2, 4, 6, 8, 10],
#     'min_samples_leaf': [1, 2, 3, 4],
#     'max_features': [None, 'sqrt', 'log2'],
#     'random_state': [30]
# }
#
# # 创建决策树回归器
# dt_reg = DecisionTreeRegressor()
#
# # 创建网格搜索对象
# grid_search = GridSearchCV(
#     estimator=dt_reg,
#     param_grid=param_grid,
#     scoring='r2',  # 使用R²作为评分标准
#     cv=5,          # 5折交叉验证
#     n_jobs=-1,     # 使用所有可用的CPU核心
#     verbose=1      # 显示进度
# )
#
# # 执行网格搜索
# grid_search.fit(x_train, y_train)
#
# # 输出最佳参数
# print("最佳参数组合: ", grid_search.best_params_)
# #特征重要性图
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