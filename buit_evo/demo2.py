import pandas as pd

data = pd.read_feather(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/单店模型预测评估/sales_qty.ft')
'''数据清洗'''
data['product_num'] =data['product_num'].astype('int')
data['order_date'] = pd.to_datetime(data['order_date'])
'''删除不对的节假日编码'''
del data['is_holiday']
holiday = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/单店模型预测评估/holiday.xlsx')
holiday['日期'] = pd.to_datetime(holiday['日期'])
'''删除东莞海德广场kkv'''
data = data.loc[data['store_num']!='V76907']

'''增加正确的节假日编码'''
data = pd.merge(data,holiday,left_on='order_date',right_on='日期',how='left')
del data['日期']
del data['season']
data['is_holiday'] = data['is_holiday'].fillna(0)

data_ca = pd.read_excel('/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/单店模型预测评估/cate.xlsx')

data_ca = data_ca[['product_num','distinct_product_sales_category']]
data_ca['product_num'] = data_ca['product_num'].astype('int')
data = pd.merge(data,data_ca,on='product_num',how='left')


data = data.loc[(data['distinct_product_sales_category'] == 'A类')|(data['distinct_product_sales_category'] == 'B类')|
(data['distinct_product_sales_category'] == 'C类')|(data['distinct_product_sales_category'] == 'E类')]

store_num = {'num':['V02303','V02901','V35103','V75502','V89101','V93101','V99103'],
             'store_numb':['2303','2901','5103','5502','9101','3101','9103']}
data_store = pd.DataFrame(store_num)


data = pd.merge(data,data_store,left_on='store_num',right_on='num',how='left')
del data['store_num']
del data['num']

data['store_numb'] = data['store_numb'].astype('int')


'''切分训练集和验证集'''
data_train = data.loc[data['order_date']<'2022-07-01']


data_test = data.loc[data['order_date']>='2022-07-01']
data_test = data_test.loc[data_test['order_date'] != '2022-07-08']
'''月份'''
data_test['month'] = data_test['order_date'].dt.month
'''星期'''
data_test['week_date'] = data_test['order_date'].dt.weekday
data_test['week_date'] = data_test['week_date']+1
'''是否为周末'''
del data_test['is_weekend']
is_week = {'date':[1,2,3,4,5,6,7],
           'is_weekend':[0,0,0,0,0,1,1]}
data_is_weekday = pd.DataFrame(is_week)
data_test = pd.merge(data_test,data_is_weekday,left_on='week_date',right_on='date',how='left')
del data_test['date']

'''季节性基础数据源'''
is_season = {'date':[1,2,3,4,5,6,7,8,9,10,11,12],
             'season':[4,4,1,1,1,2,2,2,3,3,3,4]}
data_is_season = pd.DataFrame(is_season)

'''训练集季节性'''
data_train = pd.merge(data_train,data_is_season,left_on='month',right_on='date',how='left')
del data_train['date']

'''测试集合季节性'''
data_test = pd.merge(data_test,data_is_season,left_on='month',right_on='date',how='left')
del data_test['date']

'''验证集增加天气因素'''
del data_test['max_temp']
del data_test['min_temp']
del data_test['wind']
del data_test['weather']

weather = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/单店模型预测评估/weather.xlsx')
weather['时间'] = pd.to_datetime(weather['时间'])
data_test = pd.merge(data_test,weather,left_on=['store_numb','order_date'],right_on=['编码','时间'],how='left')
del data_test['时间']
del data_test['编码']
del data_test['城市']
'''nan值替换'''
data_train.fillna(0,inplace=True)
data_test.fillna(0,inplace=True)

# print(data_train.info())
# print(data_test.info())


# print(data_train.isnull().any())
# print('--------')
# print(data_test.isnull().any())

# data_train1 = data_train.head(300000)
data_train.to_feather(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/单店模型预测评估/train_clear_2.ft')
# data_test.to_feather(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/单店模型预测评估/test_clear.ft')
data_test.to_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/单店模型预测评估/test_clear_2.xlsx')
# data_train1.to_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/单店模型预测评估/test_train_demo.xlsx')
'''模型预测'''
# from matplotlib import pyplot as plt
# import matplotlib
# plt.rcParams["font.family"] = 'Arial Unicode MS'
# matplotlib.rcParams['font.sans-serif']=['SimHei']
# matplotlib.rcParams['axes.unicode_minus']=False
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# import warnings
# import numpy as np
# import seaborn as sns
# from math import sqrt
# warnings.filterwarnings("ignore")
#
# '''删除无用指标'''
# del data_train['order_date']
# del data_test['order_date']
# '''训练集划分'''
# Y = data_train[['sales_qty_label']]
# del data_train['sales_qty_label']
# X = data_train
#
# '''验证集划分'''
# Y_t = data_test[['sales_qty_label']]
# del data_test['sales_qty_label']
# X_t = data_test
#
#
# '''数据集划分'''
# X_tarin,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=7)
# rf_res = RandomForestRegressor(n_jobs=-1,n_estimators=300,max_depth=12,min_samples_split=3)
# rf_res.fit(X_tarin,y_train)
# y_pre = rf_res.predict(X_test)
#
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y_test,y_pre)
# rmse = sqrt(mse)
#
# y_pt = rf_res.predict(X_t)
# mse1 = mean_squared_error(y_pt,Y_t)
# rmse1 = sqrt(mse1)
# y_predictions = [value for value in y_pt]
# y_data = pd.DataFrame(y_predictions,columns=['预测值'])
# y_data.to_excel(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/单店模型预测评估/预测值_国内kkv.xlsx')
# print('模型的RMSE',rmse1)
# r2 = r2_score(y_pt,Y_t)
#
