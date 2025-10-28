import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import shap

data = pd.read_csv(r"C:\.....")
data.drop(["Formula","Remark"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=320)

#GBR模型
gbr = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.01,
    max_depth=5,
    random_state=460,
    subsample = 0.7,
    min_samples_split = 10,

)
gbr.fit(x_train,y_train)
petrain = gbr.predict(x_train)
petest = gbr.predict(x_test)
r2_test = r2_score(y_test,petest)
r2_train = r2_score(y_train,petrain)
mse = mean_squared_error(y_test,petest)
print("训练集分数:",r2_train,"测试集分数：",r2_test,"MSE:",mse)
#递归消除
# model = GradientBoostingRegressor(n_estimators = 100,random_state=46)
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
# # # 获取特征排名和对应特征
# feature_names =x.columns.tolist()
# feature_ranking = rfe.ranking_
# ranked_features = [(feature_names[i], feature_ranking[i]) for i in range(len(feature_names))]# 创建排名和特征的对应关系
# ranked_features.sort(key=lambda x: x)# 根据排名对特征进行排序
# print("Feature Ranking and Names:")# 打印排名和特征的对应关系
# for feature, ranking in ranked_features:
#     print(f"{feature}: {ranking}")
#特征重要性图
# importance = gbr.feature_importances_
# features = x_train.columns
# indices = np.argsort(importance)
# plt.figure(figsize=(10, 6))
# plt.title('GBR Feature Importance')
# plt.barh(range(len(indices)), importance[indices], color='#1f77b4', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.tight_layout()
# plt.show()
#递归消除
# model = GradientBoostingRegressor(n_estimators=100, random_state=46)
#
# # 存储结果
# feature_counts = []
# test_r2 = []
# train_r2 = []
# test_mse = []
# train_mse = []
#
# # 反向循环从所有特征到1个特征
# for i in range(x_train.shape[1], 0, -1):
#     rfe = RFE(model, n_features_to_select=i, step=1).fit(x_train, y_train)
#
#     # 预测和计算指标
#     y_test_pred = rfe.predict(x_test)
#     y_train_pred = rfe.predict(x_train)
#     feature_counts.append(i)
#     test_r2.append(r2_score(y_test, y_test_pred))
#     train_r2.append(r2_score(y_train, y_train_pred))
#     test_mse.append(mean_squared_error(y_test, y_test_pred))
#     train_mse.append(mean_squared_error(y_train, y_train_pred))
#
# # 反转列表
# feature_counts = feature_counts[::-1]
# test_r2 = test_r2[::-1]
# train_r2 = train_r2[::-1]
# test_mse = test_mse[::-1]
# train_mse = train_mse[::-1]
#
# # 创建画布
# fig, ax1 = plt.subplots(figsize=(12, 7))
#
# # 左Y轴 - R² (红色实线)
# color = 'black'
# ax1.set_xlabel('Number of feature descriptors', fontsize=18, fontweight='bold')
# ax1.set_ylabel('R²', color=color, fontsize=18, fontweight='bold')
# line1 = ax1.plot(feature_counts, test_r2, 's-', color="red",
#                  linewidth=3, markersize=10, label='Test R²',
#                  markerfacecolor='white', markeredgewidth=2)
# ax1.tick_params(axis='y', labelcolor=color, labelsize=15)
# ax1.set_xticks(feature_counts)
# ax1.tick_params(axis='x', labelsize=15)
#
# # 右Y轴 - MSE (蓝色菱形)
# ax2 = ax1.twinx()
# line2 = ax2.plot(feature_counts, test_mse, 'D-', color="blue",
#                  linewidth=3, markersize=10, label='Test MSE',
#                  markerfacecolor='white', markeredgewidth=2)
# ax2.set_ylabel('MSE', color=color, fontsize=18, fontweight='bold')
# ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
#
# # 合并图例
# lines = line1 + line2
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper center',
#            bbox_to_anchor=(0.5, 1.15),
#            ncol=2, fontsize=15, frameon=False)
#
# # 调整边框
# for spine in ax1.spines.values():
#     spine.set_linewidth(1.5)
# ax1.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)
#
# plt.tight_layout()
# # plt.savefig('feature_selection_performance.png',
# #             dpi=600, bbox_inches='tight',
# #             facecolor='white')
# plt.show()
#SHAP值计算
# explainer = shap.TreeExplainer(gbr)
# shap_values = explainer.shap_values(x_test)
# # shap.summary_plot(shap_values, x_test, feature_names=x.columns)
# shap.summary_plot(shap_values, x_test, feature_names=x.columns, show=True)
# plt.tight_layout()
# plt.show()