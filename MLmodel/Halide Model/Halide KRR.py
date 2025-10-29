import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"C:\......")
data.drop(["Formula","Remark"],inplace=True,axis=1)
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=30)
#KRR模型
krr = KernelRidge(
    alpha=0.7,          # 正则化强度
    kernel='rbf',       # 核函数类型
    gamma=0.4,          # RBF核参数
    kernel_params=None
).fit(x_train,y_train)
petrain = krr.predict(x_train)
petest = krr.predict(x_test)
r2_test = r2_score(y_test,petest)
r2_train = r2_score(y_train,petrain)
print(r2_train,r2_test)
#不同和函数选择
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# results = {}
# for kernel in kernels:
#     if kernel == 'poly':
#         krr = KernelRidge(kernel=kernel, degree=3, alpha=0.1)
#     elif kernel == 'rbf':
#         krr = KernelRidge(kernel=kernel, gamma=0.1, alpha=0.1)
#     else:
#         krr = KernelRidge(kernel=kernel, alpha=0.1)
#
#     krr.fit(x_train, y_train)
#     y_pred = krr.predict(x_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     results[kernel] = {'MSE': mse, 'R2': r2}
#
# # 展示结果
# for kernel, metrics in results.items():
#     print(f"Kernel: {kernel}")
#     print(f"  MSE: {metrics['MSE']:.4f}")
#     print(f"  R2: {metrics['R2']:.4f}")
#     print()
#网格搜索
from sklearn.model_selection import GridSearchCV

# 参数网格
# param_grid = [
#     {
#         'kernel': ['rbf'],
#         'gamma': np.logspace(-2, 2, 5),
#         'alpha': np.logspace(-3, 1, 5)
#     },
#     {
#         'kernel': ['poly'],
#         'degree': [2, 3, 4],
#         'alpha': np.logspace(-3, 1, 5),
#         'coef0': [0, 1]
#     }
# ]

# 创建GridSearchCV对象
# grid_search = GridSearchCV(
#     KernelRidge(),
#     param_grid,
#     cv=5,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1,
#     verbose=1
# )
#
# # 执行网格搜索
# grid_search.fit(x_train, y_train)
#
# # 最佳参数
# print("Best Parameters:", grid_search.best_params_)
#
# # 使用最佳模型进行预测
# best_krr = grid_search.best_estimator_
# y_pred = best_krr.predict(x_test)
#
# # 评估
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Optimized Model MSE: {mse:.4f}")
# print(f"Optimized Model R-squared: {r2:.4f}")