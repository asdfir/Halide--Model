import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBRegressor as XGBR
from xgboost import plot_importance
from sklearn.feature_selection import RFE
import joblib
import shap


data = pd.read_csv(r"C:\.....",encoding='gbk')
data.drop(["Formula","Remark","a","c","MN_Mis"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)


#模型
dfull = xgb.DMatrix(x,y)
dtrain = xgb.DMatrix(x_train,y_train)
dtest = xgb.DMatrix(x_test,y_test)
param1 = {'verbosity':0
       ,"obj":"reg:linear"
       ,"subsample":0.759
       ,"max_depth":10
       ,"eta":1
       ,"gamma":0.021
       ,"lambda":1
       ,"alpha":0
       ,"colsample_bytree":1
       ,"colsample_bylevel":1
       ,"colsample_bynode":1
       ,"nfold":5}
num_round = 200

cvresult1=xgb.train(param1,dtrain,num_round)
preds = cvresult1.predict(dtest)
preds2 = cvresult1.predict(dtrain)
r2_test = r2_score(y_test,preds)
r2_train = r2_score(y_train,preds2)
mse = mean_squared_error(y_test,preds)
rmse = np.sqrt(mse)
print("训练集分数：",r2_train,"测试集分数:",r2_test,
      "MSE:",mse,"RMSE:",rmse)

#交叉验证
# param1_ = {'verbosity':0
#        ,"obj":"reg:linear"
#        ,"subsample":0.5
#        ,"max_depth":6
#        ,"eta":0.1
#        ,"gamma":0
#        ,"lambda":1
#        ,"alpha":0
#        # ,"eval_metric" : "R-squared"
#        ,"colsample_bytree":1
#        ,"colsample_bylevel":1
#        ,"colsample_bynode":1
#        ,"nfold":5}
# num_round = 100
# n_fold = 5
# cv_result = xgb.cv(param1_,dtrain,num_round,n_fold)
# print(cv_result)
# plt.figure(figsize=(20,5))
# plt.grid()
# plt.plot(range(1,101),cv_result.iloc[:,0],c="red",label="train,gamma=0")
# plt.plot(range(1,101),cv_result.iloc[:,2],c="orange",label="test,gamma=0")
# plt.legend()
# plt.show()
#RFE递归消除
# model = XGBR(verbosity=0,
#              subsample=0.759,
#              max_depth=10,
#              learning_rate=1,
#              gamma=0.021,
#              reg_lambda=1,
#              reg_alpha=0,
#              colsample_bytree=1,
#              colsample_bylevel=1,
#              colsample_bynode=1,
#              n_estimators=200,
#              random_state=42)
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
# plt.text(feature_counts[7], scores[7], str(scores[7]), ha='center', va='bottom')
# plt.text(feature_counts[7],scores2[7],str(scores2[7]),ha = "center",va = "bottom")
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
#  print(f"{feature}: {ranking}")
# 模型性能的学习曲线
# def plot_learning_curve(estimator,
#                         title,
#                         x,y,
#                         ax = None#
#                        ,ylim = None
#                        , cv = None
#                        ,n_jobs = None):
#     train_size,train_score,test_score = learning_curve(estimator,x,y,cv = cv,n_jobs = n_jobs)
#     if ax == None:
#         ax = plt.gca()
#     else:
#         ax = plt.figure()
#     ax.set_title(title)
#     if ylim is not None:
#         ax.set_ylim(*ylim)
#     ax.set_xlabel("Training examples")
#     ax.set_ylabel("Score")
#     ax.grid()
#     ax.plot(train_size,np.mean(train_score,axis = 1),"o-",c = "g",label = "Train score")
#     ax.plot(train_size,np.mean(test_score,axis = 1),"o-",c = "r",label = "Test score")
#     ax.legend(loc = "best")
#     return ax
# cv = ShuffleSplit(n_splits = 10,
#                   test_size = 0.3,#0.2*50作
#                   random_state = 0)
#
#
# plot_learning_curve(reg,"XGB",x,y,cv = cv,ax = None)
# plt.show()
#调参学习曲线
# fig,ax = plt.subplots(1,figsize=(15,10))
# ax.set_ylim(top=5)
# ax.grid()
# ax.plot(range(1,201),cvresult1.iloc[:,0],c="red",label="train,original")
# ax.plot(range(1,201),cvresult1.iloc[:,2],c="orange",label="test,original")
# ax.legend(fontsize="xx-large")
# plt.show()
#模型拟合程度
# plt.figure()
# plt.scatter(y_test,preds,marker='s',color='red',
#     edgecolor='black',
#     s=100)
# plt.scatter(y_train,preds2,marker="*",color = "blue",s = 100)
# plt.plot([y_train.min(), y_train.max()], [preds2.min(), preds2.max()],'k--', lw=2)
# plt.legend()
# plt.text(
#     x=0.01 * y.max(),
#     y=0.9 * y.max(),
#     s='Train',
#     fontsize=20,
#     color='red',
#     fontweight='bold'
# )
# plt.text(
#     x=0.01 * y.max(),
#     y=0.7 * y.max(),
#     s='Test',
#     fontsize=20,
#     color='green',
#     fontweight='bold'
# )
# plt.text(
#     x=0.01 * y.max(),
#     y=0.6 * y.max(),
#     s=f'$R^2 = {r2_test:.3f}$',
#     fontsize=20,
#     fontweight='bold'
# )
# # plt.text(
# #     x=0.01 * y.max(),
# #     y=0.66 * y.max(),
# #     s=f'$MSE = {mse:.3f}$',
# #     fontsize=12,
# # )
# plt.text(
#     x=0.01 * y.max(),
#     y=0.8 * y.max(),
#     s=f'$R^2 = {r2_train:.3f}$',
#     fontsize=20,
#     fontweight='bold'
# )
#
# plt.xlabel('Experimental ionic conductivity (mS/cm)',fontsize=18, labelpad=10,fontweight='bold')
# plt.ylabel('Predicted ionic conductivity (mS/cm)',fontsize=18, labelpad=10,fontweight='bold')
# plt.xticks(fontsize=20, fontweight='bold')
# plt.yticks(fontsize=20, fontweight='bold')
# plt.grid(False)
# plt.tight_layout()
# plt.show()
# 预测值与真实值的对比
# plt.plot(range(len(y_train)),sorted(y_train),c="black",label = "data")
# plt.plot(range(len(preds2)),sorted(preds2),c="red",label = "predict")
# plt.legend()
# plt.show()

#SHAP值计算
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical symbols
# plt.rcParams['font.size'] = 20
#
# # Initialize explainer and calculate SHAP values
# explainer = shap.TreeExplainer(cvresult1)
# shap_values = explainer.shap_values(x_test)
#
# # Create SHAP summary plot with enhanced formatting
# fig, ax = plt.subplots(figsize=(8, 6))
# shap.summary_plot(
#     shap_values,
#     x_test,
#     feature_names=x.columns,
#     plot_type="dot",  # or "bar" for compact view
#     show=False,  # Disable auto-show
#     plot_size=None,  # Use figsize instead
#     color=plt.get_cmap('viridis'),  # Color-blind friendly
#     alpha=0.7,  # Better for printed outputs
#     max_display=15,  # Limit to top features
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

# for item in ax.get_yticklabels():
#     item.set_fontsize(18)
#     item.set_fontweight('bold')

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
# # Adjust layout and save
# plt.tight_layout()
#
# plt.show()
# #SHAP值特征重要性
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical symbols
#
# explainer = shap.TreeExplainer(cvresult1)
# shap_values = explainer.shap_values(x_test)
# plt.figure(figsize=(8, 6))
# shap.summary_plot(shap_values, x_test, feature_names=x.columns, plot_type="bar", show=False)
# ax = plt.gca()

# for label in ax.get_yticklabels():
#     label.set_fontweight('bold')
#     label.set_fontsize(16)
#
# for label in ax.get_xticklabels():
#     label.set_fontweight('bold')
#     label.set_fontsize(16)

# ax.set_xlabel('mean(|SHAP value|)', fontsize=16, fontweight='bold')
#
# plt.tight_layout()
# plt.show()
#递归消除
# model = XGBR(verbosity=0,
#              subsample=0.759,
#              max_depth=10,
#              learning_rate=0.17,
#              gamma=0.021,
#              reg_lambda=1,
#              reg_alpha=0,
#              colsample_bytree=1,
#              colsample_bylevel=1,
#              colsample_bynode=1,
#              n_estimators=200,
#              random_state=42)
#
#
# feature_counts = []
# test_r2 = []
# train_r2 = []
# test_mse = []
# train_mse = []
#
#
# for i in range(x_train.shape[1], 0, -1):
#     rfe = RFE(estimator=model,
#               n_features_to_select=i,
#               step=1).fit(x_train, y_train)
#
#
#     y_test_pred = rfe.predict(x_test)
#     y_train_pred = rfe.predict(x_train)
#     feature_counts.append(i)
#     test_r2.append(r2_score(y_test, y_test_pred))
#     train_r2.append(r2_score(y_train, y_train_pred))
#     test_mse.append(mean_squared_error(y_test, y_test_pred))
#     train_mse.append(mean_squared_error(y_train, y_train_pred))
#
#
# feature_counts = feature_counts[::-1]
# test_r2 = test_r2[::-1]
# train_r2 = train_r2[::-1]
# test_mse = test_mse[::-1]
# train_mse = train_mse[::-1]
#
#
# max_r2_idx = np.argmax(test_r2)
# min_mse_idx = np.argmin(test_mse)
#
#
# fig, ax1 = plt.subplots(figsize=(12, 7))
#
#
# color = 'black'
# ax1.set_xlabel('Number of feature descriptors', fontsize=20, fontweight='bold')
# ax1.set_ylabel('R²', color=color, fontsize=20, fontweight='bold')
# line1 = ax1.plot(feature_counts, test_r2, 's-', color="red",
#                  linewidth=3, markersize=10, label='R²',
#                  markerfacecolor='white', markeredgewidth=2)
#
#
#
# ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
# ax1.set_xticks(feature_counts)
# ax1.tick_params(axis='x', labelsize=20)
#
#
# for label in ax1.get_xticklabels():
#     label.set_fontweight('bold')
# for label in ax1.get_yticklabels():
#     label.set_fontweight('bold')
#
#
# ax2 = ax1.twinx()
# line2 = ax2.plot(feature_counts, test_mse, 'D-', color="blue",
#                  linewidth=3, markersize=10, label='MSE',
#                  markerfacecolor='white', markeredgewidth=2)
#
#
# ax2.set_ylabel('MSE', color=color, fontsize=20, fontweight='bold', labelpad=15)
# ax2.tick_params(axis='y', labelcolor=color, labelsize=20)
#
# for label in ax2.get_yticklabels():
#     label.set_fontweight('bold')
#
# lines = line1 + line2
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper center',
#            bbox_to_anchor=(0.35, 1.0),
#            ncol=2, fontsize=30, frameon=False,
#            prop={'weight': 'bold', 'size': 25})
#
#
# for spine in ax1.spines.values():
#     spine.set_linewidth(1.8)
# plt.tight_layout()
#
# plt.show()


