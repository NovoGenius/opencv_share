from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import matplotlib
plt.rcParams["font.family"] = 'Arial Unicode MS'
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
data = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/xgboost预测文件.xlsx',sheet_name=1)

data['折扣力度'].fillna(1,inplace=True)
# print(data.info())
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
# del data['疫情数据_std']
# del data['天气情况']
# del data['最高温']
# del data['最低温']
# del data['风级']
X = data
X_tarin,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=7)
# print(X_test)
# 模型训练
rand_res = RandomForestRegressor(n_estimators=100,
                          criterion='mse', max_depth=None,
                          min_samples_split=2, min_samples_leaf=1,
                          min_weight_fraction_leaf=0.0,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0,
                          bootstrap=True, oob_score=False,
                          n_jobs=None, random_state=None,
                          verbose=0, warm_start=False,
                          ccp_alpha=0.0, max_samples=None)

rand_res.fit(X_tarin,y_train)
# 对测试集 进行测试
y_pre = rand_res.predict(X_test)
y_predictions = [round(value) for value in y_pre]
y_data = pd.DataFrame(y_predictions,columns=['预测值'])
print('模型未进行调优前的预测值为',y_data.预测值.sum())
print('模型未进行调优前的真实值为',y_test.销量1.sum())
# 模型未进行调优前的均方误差
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pre)
r2 = r2_score(y_test,y_pre)
print('模型未进行调优前的均方误差为',mse)
print('模型未进行调优前的拟合优度为',r2)

# 模型第一次调优
'''max_depth 决策个数调优'''

param_test1 = {'max_depth':range(1,10,1)}

gs1 = GridSearchCV(estimator=RandomForestRegressor(n_estimators=100,
                                                      criterion='mse', max_depth=None,
                                                      min_samples_split=2, min_samples_leaf=1,
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto', max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0,
                                                      bootstrap=True, oob_score=False,
                                                      n_jobs=None, random_state=None,
                                                      verbose=0, warm_start=False,
                                                      ccp_alpha=0.0, max_samples=None),
                                                    param_grid=param_test1,
                                                    scoring='r2',
                                                    cv=4)
gs1.fit(X_tarin,y_train)
model1 = gs1.best_estimator_.fit(X_tarin,y_train)
# print(gs1.best_estimator_)

y_pre1 = model1.predict(X_test)
y_predictions1 = [round(value) for value in y_pre1]
y_data1 = pd.DataFrame(y_predictions1,columns=['预测值'])
mse1 = mean_squared_error(y_test,y_pre1)
r2_1 = r2_score(y_test,y_pre1)

# print('第一次调优后的预测值为',y_data1.预测值.sum())
# print('第一次优化后的均方误差为',mse1)
# print('第一次优化后的拟合优度为',r2_1)

# 模型第二次调优
'''min_samples_split ，min_samples_leaf 叶子个数和权重调优'''

param_test2 = {'min_samples_split':range(2,10,1),'min_samples_leaf':range(1,10,1)}

gs2 = GridSearchCV(estimator=RandomForestRegressor(n_estimators=100,
                                                      criterion='mse', max_depth=None,
                                                      min_samples_split=2, min_samples_leaf=1,
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto', max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0,
                                                      bootstrap=True, oob_score=False,
                                                      n_jobs=None, random_state=None,
                                                      verbose=0, warm_start=False,
                                                      ccp_alpha=0.0, max_samples=None),
                                                    param_grid=param_test2,
                                                    scoring='r2',
                                                    cv=4)
gs2.fit(X_tarin,y_train)
model2 = gs2.best_estimator_.fit(X_tarin,y_train)
# print(gs2.best_estimator_)

y_pre2 = model2.predict(X_test)
y_predictions2 = [round(value) for value in y_pre2]
y_data2 = pd.DataFrame(y_predictions2,columns=['预测值'])
mse2 = mean_squared_error(y_test,y_pre2)
r2_2 = r2_score(y_test,y_pre2)

# print('第二次调优后的预测值为',y_data2.预测值.sum())
# print('第二次优化后的均方误差为',mse2)
# print('第二次优化后的拟合优度为',r2_2)

# 模型第三次调优
'''max_depth 决策个数调优'''

param_test3 = {'n_estimators':range(50,200,2)}

gs3 = GridSearchCV(estimator=RandomForestRegressor(n_estimators=100,
                                                      criterion='mse', max_depth=None,
                                                      min_samples_split=2, min_samples_leaf=1,
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto', max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0,
                                                      bootstrap=True, oob_score=False,
                                                      n_jobs=None, random_state=None,
                                                      verbose=0, warm_start=False,
                                                      ccp_alpha=0.0, max_samples=None),
                                                    param_grid=param_test3,
                                                    scoring='r2',
                                                    cv=4)
gs3.fit(X_tarin,y_train)
model3 = gs3.best_estimator_.fit(X_tarin,y_train)
# print(gs3.best_estimator_)

y_pre3 = model3.predict(X_test)
y_predictions3 = [round(value) for value in y_pre3]
y_data3 = pd.DataFrame(y_predictions3,columns=['预测值'])
mse3 = mean_squared_error(y_test,y_pre3)
r2_3 = r2_score(y_test,y_pre3)

# print('第三次调优后的预测值为',y_data3.预测值.sum())
# print('第三次优化后的均方误差为',mse3)
# print('第三次优化后的拟合优度为',r2_3)

def pre_data(data_pre):
    if r2 > r2_1 and r2 > r2_2 and r2 > r2_3:
        y_hat = rand_res.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    elif r2_1 > r2_2 and r2_1 > r2_3 :
        y_hat = model1.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    elif r2_2 > r2_3 :
        y_hat = model2.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    else:
        y_hat = model3.predict(data_pre)
        y_predictions_hat = [round(value) for value in y_hat]
    y_pre_last = pd.DataFrame(y_predictions_hat,columns=['预测值'])
    y_compare = y_pre_last.预测值.sum()
    y_t = y_test.销量1.sum()
    print('模型最终的预测值为', y_pre_last.预测值.sum())
    if y_t >= y_compare :
        print('模型的预测精度为',y_compare/y_t)
    else:
        print('模型的预测精度为', y_t / y_compare)
pre_data(X_test)
