import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import matplotlib
plt.rcParams["font.family"] = 'Arial Unicode MS'
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import pandas as pd
import warnings
import seaborn as sns
from math import sqrt
warnings.filterwarnings("ignore")
data = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/xgboost预测文件.xlsx')

# 数据清洗
del data['是否节假日1']
# 1.团购销量数据清洗
data_avg = data['销量'].mean()
data_std = data['销量'].std()
data['z_score'] = (data['销量'] - data_avg)/ data_std
data = data.loc[data['z_score']<=3]
data = data.drop('z_score',axis =1)
# 2.疫情数据标准化
data1_avg = data['疫情情况'].mean()
data1_std = data['疫情情况'].std()
data['疫情数据_std'] = (data['疫情情况'] - data1_avg) / data1_std
data = data.drop('疫情情况',axis =1)

# 数据集划分
Y = data[['销量']]
Y['销量1'] = round(Y['销量'])
del Y['销量']

del data['销量']
del data['日期']
del data['上市天数']
X = data
fig, ax = plt.subplots(figsize = (9,9))
sns.heatmap(X.corr(), annot = True, linewidths=.5, cmap="YlGnBu")
plt.title('Correlation between features', fontsize = 30)
plt.tight_layout()
# plt.show()


X_tarin,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=7)
# print(X_test)
# 模型训练
xgb_res = XGBRegressor()
xgb_res.fit(X_tarin,y_train)

# 对测试集 进行测试
y_pre = xgb_res.predict(X_test)
y_predictions = [round(value) for value in y_pre]
y_data = pd.DataFrame(y_predictions,columns=['预测值'])
# print('模型未进行调优前的预测值为',y_data.预测值.sum())
# print('模型未进行调优前的真实值为',y_test.销量1.sum())
# print('预测精度为',y_data.预测值.sum()/y_test.销量1.sum())
# 模型特征值
plot_importance(xgb_res)
# plt.show()

# 模型未进行调优前的均方误差
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pre)
rmse = sqrt(mse)
r2 = r2_score(y_test,y_pre)
print('模型未进行调优前的均方误差为',rmse)
# print('模型未进行调优前的拟合优度为',r2)


# 模型调优
'''第一次优化树的深度和叶子节点的权重'''

from sklearn.model_selection import GridSearchCV
param_test1 = {'max_depth':range(3,10,1),'min_child_weight':range(2,7,1)}

gsearch1 = GridSearchCV(estimator=XGBRegressor(learning_rate =0.05, n_estimators=140, max_depth=10,
                                         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'reg:squarederror',
                                         nthread=4, scale_pos_weight=1, seed=27),
                                         param_grid = param_test1, scoring='r2',n_jobs=-1, cv=5)
gsearch1.fit(X_tarin,y_train)
model1 = gsearch1.best_estimator_.fit(X_tarin,y_train)
# print(gsearch1.best_estimator_)
'''
XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.8,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.05, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=3, max_leaves=0, min_child_weight=2,
             missing=nan, monotone_constraints='()', n_estimators=140, n_jobs=4,
             nthread=4, num_parallel_tree=1, predictor='auto', random_state=27,
             reg_alpha=0, ...)

max_depth=3    min_child_weight=2
'''

y_1= model1.predict(X_test)
mse1 = mean_squared_error(y_test,y_1)
rmse1 = sqrt(mse1)
r2_1 = r2_score(y_test,y_1)
y_predictions1 = [round(value) for value in y_1]
y_data1 = pd.DataFrame(y_predictions1,columns=['预测值'])
# print('第一次优化后的预测值为',y_data1.预测值.sum())
print('第一次优化后的均方误差为',rmse1)
# print('第一次优化后的拟合优度为',r2_1)

# 第二次优化参数
'''gamma 指定了节点分裂所需的最小损失函数下降值'''
param_test2 = {'gamma':[i/100.0 for i in range(0,100)]}
gsearch2 = GridSearchCV(estimator=XGBRegressor(learning_rate =0.05, n_estimators=140, max_depth=3,
                                         min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'reg:squarederror',
                                         nthread=4, scale_pos_weight=1, seed=27),
                                         param_grid = param_test2, scoring='r2',n_jobs=-1, cv=5)

gsearch2.fit(X_tarin,y_train)
model2 = gsearch2.best_estimator_.fit(X_tarin,y_train)
# print(gsearch2.best_estimator_)
'''
XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.8,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0.22, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.05, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=3, max_leaves=0, min_child_weight=2,
             missing=nan, monotone_constraints='()', n_estimators=140, n_jobs=4,
             nthread=4, num_parallel_tree=1, predictor='auto', random_state=27,
             reg_alpha=0, ...)

gamma=0.22
'''

y_2= model2.predict(X_test)
mse2 = mean_squared_error(y_test,y_2)
rmse2 = sqrt(mse2)
r2_2 = r2_score(y_test,y_2)
y_predictions2 = [round(value) for value in y_2]
y_data2 = pd.DataFrame(y_predictions2,columns=['预测值'])
# print('第二次优化后的预测值为',y_data2.预测值.sum())
print('第二次优化后的均方误差为',rmse2)
# print('第二次优化后的拟合优度为',r2_2)

# 第三次优化
'''
参数 subsample 和 colsample_bytree 进行调整。
subsample 控制对于每棵树的随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。
但是，如果这个值设置得过小，它可能会导致欠拟合
'''
param_test3 = {'subsample':[i/10.0 for i in range(1,10)],'colsample_bytree':[i/10.0 for i in range(1,10)]}
gsearch3 = GridSearchCV(estimator=XGBRegressor(learning_rate =0.05, n_estimators=140, max_depth=3,
                                         min_child_weight=2, gamma=0.22, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'reg:squarederror',
                                         nthread=4, scale_pos_weight=1, seed=27),
                                         param_grid = param_test3, scoring='r2',n_jobs=-1, cv=5)

gsearch3.fit(X_tarin,y_train)
model3 = gsearch3.best_estimator_.fit(X_tarin,y_train)
# print(gsearch3.best_estimator_)
'''
XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.5,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0.22, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.05, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=3, max_leaves=0, min_child_weight=2,
             missing=nan, monotone_constraints='()', n_estimators=140, n_jobs=4,
             nthread=4, num_parallel_tree=1, predictor='auto', random_state=27,
             reg_alpha=0, ...)
colsample_bytree=0.5
'''
y_3 = model3.predict(X_test)
mse3 = mean_squared_error(y_test,y_3)
rmse3 = sqrt(mse3)
r2_3 = r2_score(y_test,y_3)
y_predictions3 = [round(value) for value in y_3]
y_data3 = pd.DataFrame(y_predictions3,columns=['预测值'])
# print('第三次优化后的预测值为',y_data3.预测值.sum())
print('第三次优化后的均方误差为',rmse3)
# print('第三次优化后的拟合优度',r2_3)


# 第四次优化
param_test4 = {'n_estimators': range(1, 200, 2)}
gsearch4 = GridSearchCV(estimator=XGBRegressor(learning_rate =0.05, n_estimators=140, max_depth=3,
                                         min_child_weight=2, gamma=0.22, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'reg:squarederror',
                                         nthread=4, scale_pos_weight=1, seed=27),
                                         param_grid = param_test4, scoring='r2',n_jobs=-1, cv=5)

gsearch4.fit(X_tarin,y_train)
model4 = gsearch4.best_estimator_.fit(X_tarin,y_train)
# print(gsearch4.best_estimator_)
'''
XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.8,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0.22, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.05, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=3, max_leaves=0, min_child_weight=2,
             missing=nan, monotone_constraints='()', n_estimators=83, n_jobs=4,
             nthread=4, num_parallel_tree=1, predictor='auto', random_state=27,
             reg_alpha=0, ...)
n_estimators=83
'''
y_4= model4.predict(X_test)
mse4 = mean_squared_error(y_test,y_4)
rmse4 = sqrt(mse4)
r2_4 = r2_score(y_test,y_4)
y_predictions4 = [round(value) for value in y_4]
y_data4 = pd.DataFrame(y_predictions4,columns=['预测值'])
# print('第四次优化后的预测值为',y_data4.预测值.sum())
print('第四次优化后的均方误差为',rmse4)
# print('第四次优化后的拟合优度为',r2_4)
# 第五次优化 学习率

param_test5 = {'learning_rate':[i/100.0 for i in range(1,100)]}
gsearch5 = GridSearchCV(estimator=XGBRegressor(learning_rate =0.05, n_estimators=140, max_depth=3,
                                         min_child_weight=2, gamma=0.22, subsample=0.8, colsample_bytree=0.5,
                                        objective= 'reg:squarederror',
                                         nthread=4, scale_pos_weight=1, seed=27),
                                         param_grid = param_test5, scoring='r2',n_jobs=-1, cv=5)

gsearch5.fit(X_tarin,y_train)
model5 = gsearch5.best_estimator_.fit(X_tarin,y_train)
# print(gsearch5.best_estimator_)
'''
XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.5,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0.22, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.03, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=3, max_leaves=0, min_child_weight=2,
             missing=nan, monotone_constraints='()', n_estimators=140, n_jobs=4,
             nthread=4, num_parallel_tree=1, predictor='auto', random_state=27,
             reg_alpha=0, ...)
learning_rate=0.03
'''

y_5= model5.predict(X_test)
mse5 = mean_squared_error(y_test,y_5)
rmse5 = sqrt(mse5)
r2_5 = r2_score(y_test,y_5)
y_predictions5 = [round(value) for value in y_5]
y_data5 = pd.DataFrame(y_predictions5,columns=['预测值'])
# print('第五次优化后的预测值为',y_data5.预测值.sum())
print('第五次优化后的均方误差为',rmse5)
# print('第五次优化后的拟合优度为',r2_5)
def pre_data(data_pre):
    if r2 > r2_1 and r2 > r2_2 and r2 > r2_3 and r2 > r2_4 and r2 > r2_5:
        y_hat = xgb_res.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    elif r2_1 > r2_2 and r2_1 > r2_3 and r2_1 > r2_4 and r2_1 > r2_5:
        y_hat = model1.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    elif r2_2 > r2_3 and r2_2 > r2_4 and r2_2 > r2_5:
        y_hat = model2.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    elif r2_3 > r2_4 and r2_3 > r2_5 :
        y_hat = model3.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    elif r2_4 > r2_5:
        y_hat = model4.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    else:
        y_hat = model5.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    y_pre_last = pd.DataFrame(y_predictions_hat,columns=['预测值'])
    y_compare = y_pre_last.预测值.sum()
    y_t = y_test.销量1.sum()
    # print('模型最终的预测值为', y_pre_last.预测值.sum())
    # if y_t >= y_compare:
    #     print('模型的预测精度为',y_compare/y_t)
    # else:
    #     print('模型的预测精度为', y_t / y_compare)
pre_data(X_test)

