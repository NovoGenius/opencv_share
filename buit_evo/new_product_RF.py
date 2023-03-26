import pandas as pd
data = pd.read_excel(r'/Users/luoliang/Documents/商品信息表.xlsx')

'''数据预处理'''
da_list = data['是否转正'].unique().tolist()
data['y'] = data['是否转正'].apply(lambda x:da_list.index(x))
'''训练集和测试集的划分'''
from sklearn.model_selection import train_test_split
X_train,y_train,X_test,y_test = train_test_split(X,Y,random_state=0.2)


'''模型建立'''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train)
y_pre = model.predict(X_test)
'''模型第一次检验'''
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pre)


'''模型调优'''
from sklearn.model_selection import GridSearchCV
ges1 = GridSearchCV(estimator=RandomForestClassifier())
'''模型评估'''
