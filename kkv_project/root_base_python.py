import pandas as pd
import warnings
import pydotplus
from matplotlib import pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error
import category_encoders as encoders
from matplotlib import pyplot as plt
import matplotlib
plt.rcParams["font.family"] = 'Arial Unicode MS'
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import r2_score
import warnings
import numpy as np
import seaborn as sns
from math import sqrt
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = 'Arial Unicode MS'
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
warnings.filterwarnings('ignore')
'''文件用feather格式进行压缩'''
data_idn = pd.read_feather(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/印尼单店模型预测评估/idn_sales_qty.ft')
data_idn.fillna(0,inplace=True)
data_idn['product_num'] = data_idn['product_num'].astype('object')
'''加入商品类别'''
data_cate = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/印尼单店模型预测评估/cate_idn更新大中小类.xlsx')
data_cate['商品编码'] = data_cate['商品编码'].astype('object')
data_idn = pd.merge(data_idn,data_cate,left_on='product_num',right_on='商品编码',how='left')
del data_idn['商品编码']
del data_idn['大类']
del data_idn['中类']
del data_idn['小类']

'''删除城市指标'''
del data_idn['city']
'''时间转化成datatime'''
data_idn['order_date'] = pd.to_datetime(data_idn['order_date'])

'''节假日重新编码'''
del data_idn['is_holiday']
data_holiday = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/印尼单店模型预测评估/印尼节日-2022(1).xlsx')
data_holiday['日期'] = pd.to_datetime(data_holiday['日期'])
data_idn = pd.merge(data_idn,data_holiday,left_on='order_date',right_on='日期',how='left')
del data_idn['日期']
# del data_idn['is_holiday']
del data_idn['is_holidaycode']

'''疫情数据'''
data_covid = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/印尼单店模型预测评估/疫情数据.xlsx')
'''用前后一天的数据进行均值填充'''
def foo(row):
    if any(row.isna()):
        data_covid.loc[row.name,row.isna()] = data_covid.expanding().mean().shift(-1).loc[row.name,:]
data_covid.apply(foo,axis=1)
data_covid['日期'] = pd.to_datetime(data_covid['日期'])
data_idn = pd.merge(data_idn,data_covid,left_on='order_date',right_on='日期',how='left')
del data_idn['日期']

'''印尼week转码'''
data_idn['week_date'] = data_idn['order_date'].dt.weekday
data_idn['week_date'] = data_idn['week_date'] + 1

'''增加周末编码'''
del data_idn['is_weekend']
is_week = {'date':[1,2,3,4,5,6,7],
           'is_weekend':[0,0,0,0,0,1,1]}
data_is_weekday = pd.DataFrame(is_week)
data_idn = pd.merge(data_idn,data_is_weekday,left_on='week_date',right_on='date',how='left')
del data_idn['date']

'''字符串转化成浮点数'''
data_idn['max_temp'] = data_idn['max_temp'].astype('float')
data_idn['min_temp'] = data_idn['min_temp'].astype('float')
data_idn['wind'] = data_idn['wind'].astype('float')

'''星期转化成object'''
data_idn['week_date'] = data_idn['week_date'].astype('object')

data_idn = data_idn.loc[data_idn['商品类别'] == 'A类']
data_idn = data_idn.reset_index(drop=True)
data_idn.fillna(0,inplace=True)
del data_idn['商品类别']

'''店铺编码'''
# store = {'store':['IDNV001','IDNV002','IDNV003','IDNV004','IDNV005','IDNV006'],
#          'store_numb':[1,2,3,4,5,6]}
# data_store = pd.DataFrame(store)
# data_idn = pd.merge(data_idn,data_store,left_on='store_num',right_on='store',how='left')
# del data_idn['store']
# del data_idn['store_num']


'''删除前一天销量'''
del data_idn['sales_qty_label']
# del data_idn['sales_qty_t_1']

'''删除天气因素'''
# del data_idn['max_temp']
# del data_idn['min_temp']
# del data_idn['wind']
# del data_idn['weather']
# del data_idn['season']
'''划分训练集和测试集'''

data_train = data_idn.loc[data_idn['order_date'] < '2022-07-01']
data_test = data_idn.loc[data_idn['order_date'] >= '2022-07-01']
data_test = data_test.loc[data_test['order_date'] != '2022-07-08']
data_test1 = data_test


# # print(data_test.info())
# # print(data_train.info())
# # print(data_test.head())
# # data_test.to_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/印尼单店模型预测评估/test.xlsx')

data_train.fillna(0,inplace=True)
data_test.fillna(0,inplace=True)
data_test1.fillna(0,inplace = True)

data_test = data_test.reset_index(drop=True)
data_test1 = data_test1.reset_index(drop=True)

# print(data_test1)
'''数据进行重新编码'''
# data_test['week_date'] = data_test['week_date'].astype('object')
# data_test['out_of_stock_num'] = data_test['out_of_stock_num'].astype('object')
# data_test['month'] = data_test['month'].astype('object')
#
#
# data_train['week_date'] = data_train['week_date'].astype('object')
# data_train['out_of_stock_num'] = data_train['out_of_stock_num'].astype('object')
# data_train['month'] = data_train['month'].astype('object')

'''删除无用指标'''
del data_train['order_date']
del data_test['order_date']
'''训练集划分'''
Y = data_train[['sales_qty']]
del data_train['sales_qty']
X = data_train

'''验证集划分'''
Y_t = data_test[['sales_qty']]
del data_test['sales_qty']
X_t = data_test
'''使用encoders 对于产品编码和店铺编码进行转码'''
'''训练集'''
enc = encoders.TargetEncoder()
oth = enc.fit_transform(X,Y)
X1 = enc.transform(X)
'''验证集'''
X_t1 = enc.transform(X_t)

# X_t1.to_csv(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/印尼单店模型预测评估/数据转化.csv')
'''数据集划分'''
X_tarin,X_test,y_train,y_test = train_test_split(X1,Y,test_size=0.2)
# rf_res = RandomForestRegressor(n_jobs=-1,n_estimators=350,max_depth=13,min_samples_split=2,min_samples_leaf=17,
#                                max_features=0.6)
# rf_res.fit(X_tarin,y_train)
# y_pre = rf_res.predict(X_test)

'''调整n_estimators参数'''

# score_1 = []
# for i in range(10,400,20):
#     m1 = RandomForestRegressor(n_estimators=i,n_jobs=-1,random_state=77)
#     score = cross_val_score(m1,X_tarin,y_train,cv=5).mean()
#     score_1.append([i,score])
# score_1 = np.array(score_1)
# max_score = np.where(score_1==max(score_1[:-1]))[0][0]
# plt.plot(score_1[:,0],score_1[:,1])
# plt.show()

#
rmse_1 = []
features_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in features_list:
    model = RandomForestRegressor(n_jobs=-1,n_estimators=350,max_depth=13,min_samples_split=2,min_samples_leaf=17,
                                  max_features=i)
    model.fit(X_tarin,y_train)
    y_pt = model.predict(X_t1)
    rmse = mean_squared_error(y_pt,Y_t,squared=False)
    rmse_1.append(rmse)
print(rmse_1)
plt.plot(rmse_1)
plt.show()

# rmse_2 = []
# depth_list = [5,6,7,8,9,10,11,12,13,14]
# for i in depth_list:
#     model = RandomForestRegressor(n_jobs=-1,n_estimators=350,max_depth=i,min_samples_split=2,min_samples_leaf=17,
#                                   max_features=0.7)
#     model.fit(X_tarin,y_train)
#     y_pt = model.predict(X_t1)
#     rmse = mean_squared_error(y_pt,Y_t,squared=False)
#     rmse_2.append(rmse)
# print(rmse_2)
# plt.plot(rmse_2)
# plt.show()

# rmse_3 = []
# samples_list = [10,11,12,13,14,15,16,17,18,19]
# for i in samples_list:
#     model = RandomForestRegressor(n_jobs=-1,n_estimators=50,max_features=0.7,max_depth=5,min_samples_leaf=i)
#     model.fit(X_tarin,y_train)
#     y_pt = model.predict(X_t1)
#     rmse = mean_squared_error(y_pt,Y_t,squared=False)
#     rmse_3.append(rmse)
# print(rmse_3)
# plt.plot(rmse_3)
# plt.show()

# rmse_4 = []
# split_list = [2,3,4,5,6,7,8,9,10,11,12]
# for i in split_list:
#     model = RandomForestRegressor(n_jobs=-1,n_estimators=50,max_features=0.7,max_depth=5,min_samples_leaf=17,min_samples_split=i)
#     model.fit(X_tarin,y_train)
#     y_pt = model.predict(X_t1)
#     rmse = mean_squared_error(y_pt,Y_t,squared=False)
#     rmse_4.append(rmse)
# print(rmse_4)
# plt.plot(rmse_4)
# plt.show()


# rmse_5 = []
# estimators_list = [250,300,350,400]
# for i in estimators_list:
#     model = RandomForestRegressor(n_jobs=-1,n_estimators=i,max_features=0.7,max_depth=5,min_samples_leaf=17,
#                                   min_samples_split=2)
#     model.fit(X_tarin,y_train)
#     y_pt = model.predict(X_t1)
#     rmse = mean_squared_error(y_pt,Y_t,squared=False)
#     rmse_5.append(rmse)
# print(rmse_5)
# plt.plot(rmse_5)
# plt.show()

'''使用均方误差作为评估指标'''

mse = mean_squared_error(y_test,y_pre)
rmse = sqrt(mse)
print('测试集的RMSE',rmse)

'''预测验证集'''
y_pt = rf_res.predict(X_t1)
mse1 = mean_squared_error(y_pt,Y_t)
rmse1 = sqrt(mse1)
print('模型总体的RMSE',rmse1)

# '''将预测值变成dataframe'''
# y_predictions = [value for value in y_pt]
# y_data = pd.DataFrame(y_predictions,columns=['预测值'])
#
#
# # y_data.to_csv(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/印尼单店模型预测评估/预测值.csv')
#
# '''横向拼接需要把 索引重置 不然会出现拼接移位'''
# y_3 = pd.concat([data_test1,y_data],axis=1)
# # y_3.to_csv(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/印尼单店模型预测评估/预测组合.csv')
#
#
# '''画出重要特征'''
# importances_values = rf_res.feature_importances_
# importances = pd.DataFrame(importances_values, columns=["importance"])
# feature_data = pd.DataFrame(X_tarin.columns, columns=["feature"])
# importance = pd.concat([feature_data, importances], axis=1)
# importance = importance.sort_values(["importance"], ascending=True)
# importance["importance"] = (importance["importance"] * 1000).astype(int)
# importance = importance.sort_values(["importance"])
# importance.set_index('feature', inplace=True)
# importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(8, 8))
# plt.show()

# import graphviz
# from  sklearn import  tree
# i=0
# for rf in rf_res.estimators_:
#     dot_data = tree.export_graphviz(rf,out_file=None,
#                                     feature_names=X_tarin.columns,
#                                     class_names=Y.columns)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     i +=1
#     graph.write_pdf(str(i)+'DTree.pdf')
