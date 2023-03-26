import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.rcParams["font.family"] = 'Arial Unicode MS'
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
warnings.filterwarnings('ignore')
path=r"/Users/luoliang/Desktop/商品信息表(1).xlsx"
data=pd.read_excel(path,sheet_name=0)
'''数据预处理'''
dict1 = {'是否转正':['新品转正常','新品转淘汰'],'y':[1,0]}
data1 = pd.DataFrame(dict1)
data2 = pd.merge(data,data1,on='是否转正',how='left')

'''模型的Y变量'''
Y = data2[['y']]
# print(Y)

del data2['y']
del data2['是否转正']
del data2['商品名称']
del data2['商品编码']

data3 = data2.loc[data2['保质期单位'] == '年']
data3['保质期'] = data3['保质期']*12

data4 = data2.loc[data2['保质期单位'] == '月']

data_total = pd.concat([data3,data4],axis=0)


data_total['price'] = data_total['零售价'] / 2000
bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,100,110,120,130,150,180,200,1000]
data_total['price_bins']  = pd.cut(data_total['price'],bins=bins)

data_total['profit'] = (data_total['price'] - data_total['不含税成本价'])/data_total['price']

bins1  = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1]
data_total['profit_bins'] = pd.cut(data_total['profit'],bins=bins1)

bins2 =[0,6,12,18,24,30,36,42,54,78,100,150,200,1000]
data_total['保质期_bins'] = pd.cut(data_total['保质期'],bins=bins2)

# print(data_total)

del data_total['保质期单位']
del data_total['零售价']
del data_total['price']
del data_total['profit']
del data_total['不含税成本价']
del data_total['保质期']
# del data_total['商品开发']
data_clear = pd.get_dummies(data_total)

X  = data_clear
# print(X.count())
# print(Y.count())
from sklearn.model_selection import train_test_split
'''划分训练集和测试集'''
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=7)
'''导入估计器'''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_features=5)
model.fit(X_train,y_train)

y_pre = model.predict(X_test)
y1 = [value for value in y_pre ]
data8 = pd.DataFrame(y1,columns=['预测值'])

# 特征筛选
features=data_clear.columns
importances = model.feature_importances_
importances = importances[:10]
indices = np.argsort(importances)
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

# print(data8)
# print(y_test)
# data_pre = pd.concat([data8,y_test],axis=1)

# data_pre.to_excel(r'C:\Users\admin\Desktop\22.xlsx')

from sklearn.metrics import recall_score,accuracy_score,f1_score
score_recall = recall_score(y_pre,y_test)
score_acc = accuracy_score(y_pre,y_test)
score_f1 = f1_score(y_pre,y_test)

print('未调优前召回率',score_recall)
print('未调优前准确率',score_acc)
print('未调优前F值',score_f1)

from sklearn.model_selection import GridSearchCV

# data_clear.to_excel(r'C:\Users\admin\Desktop\111.xlsx')

# 模型调优
'''第一次优化树的深度和叶子节点的权重'''

# from sklearn.model_selection import GridSearchCV
# param_test1 = {'max_depth':range(3,20,1)}
#
# gsearch1 = GridSearchCV(estimator=RandomForestClassifier(max_depth=2,min_samples_split=2,
#                           min_weight_fraction_leaf=0,max_features="auto",n_jobs=-1)
#                         ,param_grid=param_test1,cv=5,scoring='roc_auc')
#
#
# gsearch1.fit(X_train,y_train)
# # print(gsearch1.best_estimator_)
#
# '''
# RandomForestClassifier(max_depth=9, min_weight_fraction_leaf=0)
# '''
#
# param_test2 = {'n_estimators':range(100,400,10)}
#
# gsearch2 = GridSearchCV(estimator=RandomForestClassifier(max_depth=9,min_samples_split=2,
#                          min_weight_fraction_leaf=0,max_features="auto",n_jobs=-1)
#                         ,param_grid=param_test2,cv=5,scoring='roc_auc')
#
# gsearch2.fit(X_train,y_train)
# # print(gsearch2.best_estimator_)
#
# '''
# RandomForestClassifier(max_depth=9, min_weight_fraction_leaf=0,
#                        n_estimators=170)
# '''
#
# parm_test3 = {'min_samples_split':range(2,20,2),'min_samples_leaf':range(2,10,1)}
#
# ges3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=170,max_depth=9,min_samples_split=2,
#         min_weight_fraction_leaf=0,max_features="auto"),param_grid=parm_test3,scoring='roc_auc',cv=5)
# ges3.fit(X_train,y_train)
# # print(ges3.best_estimator_)
#
# '''
# RandomForestClassifier(max_depth=9, min_samples_leaf=2,
#                        min_weight_fraction_leaf=0, n_estimators=170)
# '''
#
# clf = RandomForestClassifier(max_depth=9,min_samples_split=2
#                              ,min_samples_leaf=2,n_estimators=170,n_jobs=-1)
#
# clf.fit(X_train,y_train)
# clf_pred  =clf.predict(X_test)
# recall = recall_score(y_pre,y_test)
# acc = accuracy_score(y_pre,y_test)
# f1 = f1_score(y_pre,y_test)
#
# print('调优后召回率',recall)
# print('调优后准确率',acc)
# print('调优后F值',f1)

