import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import shap
import joblib  

data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","a","c"],inplace=True,axis=1)
# data["a"] =data["a"].fillna(data["a"].mean())
# data["c"] =data["c"].fillna(data["c"].mean())
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)

#GBR模型
gbr = GradientBoostingRegressor(
    n_estimators=150,  # 树的数量
    learning_rate=0.1,  # 学习率
    max_depth=7,  # 每棵树的最大深度
    random_state=31,
    subsample=0.3
)
gbr.fit(x_train,y_train)
petrain = gbr.predict(x_train)
petest = gbr.predict(x_test)
r2_test = r2_score(y_test,petest)
r2_train = r2_score(y_train,petrain)
mse = mean_squared_error(y_test,petest)
rmse = np.sqrt(mse)
# joblib.dump(gbr, 'xingbr_model.joblib')
print("训练集分数:",r2_train,"测试集分数：",r2_test,"MSE:",mse,rmse)

# x_train_reset = x_train.reset_index(drop=True)
# x_test_reset = x_test.reset_index(drop=True)
# y_train_reset = pd.Series(y_train.values, name='实际值')
# y_test_reset = pd.Series(y_test.values, name='实际值')
#
# # 创建训练集预测结果DataFrame
# train_results = pd.DataFrame({
#     '预测值': petrain
# })
#
# # 创建测试集预测结果DataFrame
# test_results = pd.DataFrame({
#     '预测值': petest
# })
#
# # 合并特征数据和预测结果
# train_combined = pd.concat([x_train_reset, y_train_reset, train_results], axis=1)
# test_combined = pd.concat([x_test_reset, y_test_reset, test_results], axis=1)
#
# # 添加数据集标识
# train_combined['数据集'] = '训练集'
# test_combined['数据集'] = '测试集'
#
# # 合并训练集和测试集
# final_results = pd.concat([train_combined, test_combined], ignore_index=True)
#
# # 导出到Excel
# output_path = r"C:\Users\小华\Desktop\GBR预测结果.xlsx"
# final_results.to_excel(output_path, index=False, encoding='gbk')
#
# print(f"预测结果已导出到: {output_path}")
# print(f"总记录数: {len(final_results)} (训练集: {len(y_train)}, 测试集: {len(y_test)})")
#递归消除
# model = GradientBoostingRegressor( n_estimators=150,  # 树的数量
#     learning_rate=0.1, # 学习率
#     max_depth=7,       # 每棵树的最大深度
#     random_state=100,
#     subsample = 0.3)
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
# print(scores2)
# print(scores)
# # 获取特征排名和对应特征
# feature_names =x.columns.tolist()
# feature_ranking = rfe.ranking_
# ranked_features = [(feature_names[i], feature_ranking[i]) for i in range(len(feature_names))]# 创建排名和特征的对应关系
# ranked_features.sort(key=lambda x: x)# 根据排名对特征进行排序
# print("Feature Ranking and Names:")# 打印排名和特征的对应关系
# for feature, ranking in ranked_features:
#     print(f"{feature}: {ranking}")
# #特征重要性图
# importance = gbr.feature_importances_
# features = x_train.columns
# indices = np.argsort(importance)
# plt.figure(figsize=(10, 6))
# bar_height = 0.5
# plt.title('50 GBR Feature Importance')
# plt.barh(range(len(indices)), importance[indices], color='#1f77b4', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.tight_layout()
# plt.show()
#特征重要性图（修改）
#importance = gbr.feature_importances_
# features = x_train.columns
# indices = np.argsort(importance)
#
# # 设置全局字体
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False
#
# # 创建图形（调整figsize宽度为8缩小整体尺寸）
# plt.figure(figsize=(8, 6))  # 原10改为8，减小宽度
#
# # 调整柱形图粗细：通过减小height参数
# bar_height = 0.7  # 默认0.8，减小此值使柱子变细
# plt.barh(range(len(indices)),
#          importance[indices],
#          height=bar_height,  # 控制柱子粗细
#          color='#1f77b4',
#          align='center')
#
# # 设置标签和标题（保持字体大小）
# # plt.title('GBR Feature Importance', fontsize=15)
# plt.yticks(range(len(indices)),
#            [features[i] for i in indices],
#            fontsize=12)  # y轴标签字体略小于标题
# plt.xlabel('Relative Importance', fontsize=13)
#
# # 调整边距使布局更紧凑
# plt.tight_layout(pad=2)  # 减小pad值使边缘更紧凑
#
# # 隐藏右边和上边的边框线
# ax = plt.gca()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('三元GBR特征重要性.png',
#             dpi=600,
#             bbox_inches='tight',
#             facecolor='white')  # 确保背景为白色
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
#     x=0.01 * y.max(),   # 与R²同一列
#     y=0.9 * y.max(),    # 向下偏移一定位置
#     s='Train',  # 换行符\n分隔多行
#     fontsize=12,
#     color='red',
# fontfamily='Times New Roman'
# )
# plt.text(
#     x=0.01 * y.max(),   # 与R²同一列
#     y=0.74 * y.max(),    # 向下偏移一定位置
#     s='Test',  # 换行符\n分隔多行
#     fontsize=12,
#     color='green',
# fontfamily='Times New Roman'
# )
# plt.text(
#     x=0.01 * y.max(),  # 文本位置（x坐标）
#     y=0.66 * y.max(),   # 文本位置（y坐标）
#     s=f'$R^2 = {r2_test:.3f}$',  # 显示R²（保留2位小数）
#     fontsize=12,
# fontfamily='Times New Roman'
# )
# # plt.text(
# #     x=0.01 * y.max(),  # 文本位置（x坐标）
# #     y=0.66 * y.max(),   # 文本位置（y坐标）
# #     s=f'$MSE = {mse:.3f}$',  # 显示R²（保留2位小数）
# #     fontsize=12,
# # )
# plt.text(
#     x=0.01 * y.max(),  # 文本位置（x坐标）
#     y=0.82 * y.max(),   # 文本位置（y坐标）
#     s=f'$R^2 = {r2_train:.3f}$',  # 显示R²（保留2位小数）
#     fontsize=12,
# fontfamily='Times New Roman'
# )
#
# plt.xlabel('true value',fontsize=14,fontfamily='Times New Roman')
# plt.ylabel('predicted value',fontsize=14,fontfamily='Times New Roman')
# # plt.title("Model Fit Effect")
# plt.grid(False)
# plt.savefig('新三元GBR模型拟合图.png',
#             dpi=600,
#             bbox_inches='tight',
#             facecolor='white')  # 确保背景为白色
# plt.show()
#新版递归消除
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

model = GradientBoostingRegressor(n_estimators=150,  #树的数量
    learning_rate=0.1, # 学习率
    max_depth=7,       # 每棵树的最大深度
    random_state=100,
    subsample = 0.3)

# 存储结果
feature_counts = []
test_r2 = []
train_r2 = []
test_mse = []
train_mse = []

# 反向循环从所有特征到1个特征
for i in range(x_train.shape[1], 0, -1):
    rfe = RFE(model, n_features_to_select=i, step=1).fit(x_train, y_train)

    # 预测和计算指标
    y_test_pred = rfe.predict(x_test)
    y_train_pred = rfe.predict(x_train)
    feature_counts.append(i)
    test_r2.append(r2_score(y_test, y_test_pred))
    train_r2.append(r2_score(y_train, y_train_pred))
    test_mse.append(mean_squared_error(y_test, y_test_pred))
    train_mse.append(mean_squared_error(y_train, y_train_pred))

# 反转列表
feature_counts = feature_counts[::-1]
test_r2 = test_r2[::-1]
train_r2 = train_r2[::-1]
test_mse = test_mse[::-1]
train_mse = train_mse[::-1]

# 创建画布
fig, ax1 = plt.subplots(figsize=(12, 7))

# 左Y轴 - R² (红色实线)
color = 'black'
ax1.set_xlabel('Number of feature descriptors', fontsize=18, fontweight='bold')
ax1.set_ylabel('R²', color=color, fontsize=18, fontweight='bold')
line1 = ax1.plot(feature_counts, test_r2, 's-', color="red",
                 linewidth=3, markersize=10, label='Test R²',
                 markerfacecolor='white', markeredgewidth=2)
ax1.tick_params(axis='y', labelcolor=color, labelsize=15)
ax1.set_xticks(feature_counts)
ax1.tick_params(axis='x', labelsize=15)

# 右Y轴 - MSE (蓝色菱形)
ax2 = ax1.twinx()
line2 = ax2.plot(feature_counts, test_mse, 'D-', color="blue",
                 linewidth=3, markersize=10, label='Test MSE',
                 markerfacecolor='white', markeredgewidth=2)
ax2.set_ylabel('MSE', color=color, fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color, labelsize=15)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center',
           bbox_to_anchor=(0.5, 1.15),
           ncol=2, fontsize=15, frameon=False)
# 调整边框
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
# plt.savefig('feature_selection_performance.png',
#             dpi=600, bbox_inches='tight',
#             facecolor='white')
plt.show()
#SHAP值计算
# explainer = shap.TreeExplainer(gbr)
# shap_values = explainer.shap_values(x_test)
# # shap.summary_plot(shap_values, x_test, feature_names=x.columns)
# shap.summary_plot(shap_values, x_test, feature_names=x.columns, show=True)  # 不显示图像
# plt.tight_layout()#自动调整图片尺寸
# # plt.savefig('shap_summary_plot.png', bbox_inches='tight', dpi=300)  # 保存为 PNG
# plt.show()

