import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import shap

data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","roce"],inplace=True,axis=1)
# data["a"] =data["a"].fillna(data["a"].mean())
# data["c"] =data["c"].fillna(data["c"].mean())
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=320)

#GBR模型
gbr = GradientBoostingRegressor(
    n_estimators=120,  # 树的数量
    learning_rate=0.02,# 学习率
    max_depth=5,       # 每棵树的最大深度
    random_state=46,
    subsample = 1,
    min_samples_split = 2,

)
gbr.fit(x_train,y_train)
petrain = gbr.predict(x_train)
petest = gbr.predict(x_test)
r2_train = r2_score(y_train,petrain)
r2_test = r2_score(y_test,petest)
mse = mean_squared_error(y_test,petest)

print("训练集分数:",r2_train,"测试集分数：",r2_test,mse)

#递归消除
# model = GradientBoostingRegressor(   n_estimators=120,  # 树的数量
#     learning_rate=0.05,# 学习率
#     max_depth=5,       # 每棵树的最大深度
#     random_state=46,
#     subsample = 1,
#     min_samples_split = 2,)
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
# # plt.text(feature_counts[7], scores[7], str(scores[7]), ha='center', va='bottom')
# # plt.text(feature_counts[7],scores2[7],str(scores2[7]),ha = "center",va = "bottom")
# plt.legend()
# print(scores2)
# print(scores)
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
#
# plt.figure(figsize=(8, 6))
#

# bar_height = 0.7
# plt.barh(range(len(indices)),
#          importance[indices],
#          height=bar_height,
#          color='#1f77b4',
#          align='center')
#

# # plt.title('GBR Feature Importance', fontsize=15)
# plt.yticks(range(len(indices)),
#            [features[i] for i in indices],
#            fontsize=12)
# plt.xlabel('Relative Importance', fontsize=13)
#

# plt.tight_layout(pad=2)
#

# ax = plt.gca()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.show()
#模型拟合程度
# plt.figure()
# plt.scatter(y_test,petest,marker='s',color='red',
#     edgecolor='black', # 正方形边框颜色
#     s=100)
# plt.scatter(y_train,petrain,marker="*",color = "blue",s = 100)
# plt.plot([y_train.min(), y_train.max()], [petrain.min(), petrain.max()],'k--', lw=2)
# plt.legend()
# plt.text(
#     x=0.07 * y.max(),
#     y=0.92 * y.max(),
#     s='Train',
#     fontsize=12,
#     color='red',fontfamily='Times New Roman'
# )
# plt.text(
#     x=0.07 * y.max(),
#     y=0.76 * y.max(),
#     s='Test',
#     fontsize=12,
#     color='green',fontfamily='Times New Roman'
# )
# plt.text(
#     x=0.07 * y.max(),
#     y=0.68 * y.max(),
#     s=f'$R^2 = {r2_test:.3f}$',
#     fontsize=12,fontfamily='Times New Roman'
# )
# # plt.text(
# #     x=0.01 * y.max(),
# #     y=0.66 * y.max(),
# #     s=f'$MSE = {mse:.3f}$',
# #     fontsize=12,
# # )
# plt.text(
#     x=0.07 * y.max(),
#     y=0.84 * y.max(),
#     s=f'$R^2 = {r2_train:.3f}$',
#     fontsize=12,fontfamily='Times New Roman'
# )
#
# plt.xlabel('true value',fontfamily='Times New Roman',fontsize=15)
# plt.ylabel('predicted value',fontfamily='Times New Roman',fontsize=15)
# # plt.title("Model Fit Effect")
# plt.grid(False)
# plt.show()
#新版递归消除
# model = GradientBoostingRegressor(   n_estimators=120,
#    learning_rate=0.05,
#  max_depth=5,
# random_state=46,
# subsample=1,
#   min_samples_split=2)

# feature_counts = []
# test_r2 = []
# train_r2 = []
# test_mse = []
# train_mse = []
#
# for i in range(x_train.shape[1], 0, -1):
#     rfe = RFE(model, n_features_to_select=i, step=1).fit(x_train, y_train)
#     y_test_pred = rfe.predict(x_test)
#     y_train_pred = rfe.predict(x_train)
#     feature_counts.append(i)
#     test_r2.append(r2_score(y_test, y_test_pred))
#     train_r2.append(r2_score(y_train, y_train_pred))
#     test_mse.append(mean_squared_error(y_test, y_test_pred))
#     train_mse.append(mean_squared_error(y_train, y_train_pred))
# feature_counts = feature_counts[::-1]
# test_r2 = test_r2[::-1]
# train_r2 = train_r2[::-1]
# test_mse = test_mse[::-1]
# train_mse = train_mse[::-1]

# fig, ax1 = plt.subplots(figsize=(12, 7))
# color = 'black'
# ax1.set_xlabel('Number of feature descriptors', fontsize=18, fontweight='bold')
# ax1.set_ylabel('R²', color=color, fontsize=18, fontweight='bold')
# line1 = ax1.plot(feature_counts, test_r2, 's-', color="red",
#                  linewidth=3, markersize=10, label='Test R²',
#                  markerfacecolor='white', markeredgewidth=2)
# ax1.tick_params(axis='y', labelcolor=color, labelsize=15)
# ax1.set_xticks(feature_counts)
# ax1.tick_params(axis='x', labelsize=15)
# ax2 = ax1.twinx()
# line2 = ax2.plot(feature_counts, test_mse, 'D-', color="blue",
#                  linewidth=3, markersize=10, label='Test MSE',
#                  markerfacecolor='white', markeredgewidth=2)
# ax2.set_ylabel('MSE', color=color, fontsize=18, fontweight='bold')
# ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
# lines = line1 + line2
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper center',
#            bbox_to_anchor=(0.5, 1.15),
#            ncol=2, fontsize=15, frameon=False)

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
#SHAP图
# explainer = shap.TreeExplainer(gbr)
# shap_values = explainer.shap_values(x_test)
#
# # Create SHAP summary plot with enhanced formatting
# fig, ax = plt.subplots(figsize=(8, 6))
# shap.summary_plot(
#     shap_values,
#     x_test,
#     feature_names=x.columns,
#     plot_type="dot",
#     show=False,
#     plot_size=None,
#     color=plt.get_cmap('viridis'),
#     alpha=0.7,
#     max_display=15
# )
#
# for collection in ax.collections:
#     if hasattr(collection, 'get_sizes'):
#
#         collection.set_sizes([40])
#
# # Enhance plot aesthetics
# ax.set_xlabel('SHAP value ', fontsize=18, fontweight='bold')
# ax.tick_params(axis='both', which='major', labelsize=13)
#
#
# for item in ax.get_yticklabels():
#     item.set_fontsize(18)
#     item.set_fontweight('bold')
#
#
# for item in ax.get_xticklabels():
#     item.set_fontsize(18)
#     item.set_fontweight('bold')
#
# all_texts = fig.findobj(match=plt.Text)
# for text in all_texts:
#     text_content = text.get_text().lower()
#     if any(keyword in text_content for keyword in ['high', 'low', 'feature']):
#         text.set_fontsize(18)
#         text.set_fontweight('bold')
#
# plt.tight_layout()
# plt.show()
# SAHP值特征重要性
# explainer = shap.TreeExplainer(gbr)
# shap_values = explainer.shap_values(x_test)
#
# plt.figure(figsize=(8.3, 6.5))
# shap.summary_plot(shap_values, x_test, feature_names=x.columns, plot_type="bar", show=False)
# ax = plt.gca()
#

# for label in ax.get_yticklabels():
#     label.set_fontweight('bold')
#     label.set_fontsize(16)
# for label in ax.get_xticklabels():
#     label.set_fontweight('bold')
#     label.set_fontsize(16)
# ax.set_xlabel('mean(|SHAP value|)', fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()
