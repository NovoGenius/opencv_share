from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from  sklearn import metrics
from sklearn.model_selection import cross_val_score,GridSearchCV
from  sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# 创建Flask对象app并初始化
app = Flask(__name__)
# 通过python装饰器的方法定义路由地址
@app.route("/")
# 定义方法 用jinjia2引擎来渲染页面，并返回一个index.html页面
def root():
    return render_template("index.html")


@app.route('/', methods=['GET', 'POST'])
# 定义app在8080端口运行
def main():
     #由于POST、GET获取数据的方式不同，需要使用if语句进行判断
     if request.method == "POST":
         gender = request.form.get("gender")
         age = request.form.get("age")
         compete = request.form.get("compete")
         income = request.form.get("income")
         home = request.form.get("home")
         new1 = {'性别系数': [gender], '年龄系数': [age], '竞争等级': [compete], '收入等级': [income], '家庭比例': [home]}
         data_new1 = pd.DataFrame(new1, index=None)


         data = pd.read_excel(r'/Users/luoliang/Desktop/商圈影响决策树版本.xlsx')
         x = pd.DataFrame(data.iloc[:, 1:6])
         data2 = pd.DataFrame(data.iloc[:, 6:])
         list2 = []
         for i in range(1, len(data2.columns) + 1):
             y = data2.iloc[:, i - 1:i]
             X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.5, random_state=0)
             model = RandomForestClassifier(n_estimators=100,
                                            n_jobs=-1,
                                            max_depth=4,
                                            bootstrap=True
                                            )
             model.fit(X_train, Y_train)
             score = model.score(X_test, Y_test)
             # print(score)
             # print('模型得分准确率：%.2f%%'%(score*100))
             y_pre = model.predict(X_test)
             # print(y_pre)
             # 样本的概率
             # print(model.predict_proba(X_test)[:,:])
             # 各个x的重要性
             importances = model.feature_importances_
             std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
             indies = np.argsort(importances)[::-1]
             # print('Feature ranking')
             # for f in range(min(20,X_train.shape[1])):
             # print('%2d) %-*s %f' % (f + 1,30,X_train.columns[indies[f]],importances[indies[f]]))
             plt.figure()
             plt.title('Feature importances')
             plt.bar(range(X_train.shape[1]), importances[indies], color='r', yerr=std[indies], align='center')
             plt.xticks(range(X_train.shape[1]), indies)
             plt.xlim([-1, X_train.shape[1]])
             # plt.show()

             f, ax = plt.subplots(figsize=(7, 5))
             ax.bar(range(len(model.feature_importances_)), model.feature_importances_)
             ax.set_title('Feature importances')
             # plt.show()
             # ROC得分
             roc = metrics.roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
             # print('roc准确率：%.2f%%'%(roc*100))
             # 画roc曲线
             pre_vali = model.predict_proba(X_test)[:, 1]
             fpr, tpr, _ = metrics.roc_curve(Y_test, pre_vali)
             roc_auc = metrics.auc(fpr, tpr)
             plt.title('roc validation')
             plt.plot(fpr, tpr, "b", label='AUC=%0.2f' % roc_auc)
             plt.legend(loc='lower right')
             plt.plot([0, 1], [0, 1], 'r--')
             plt.xlim([0, 1])
             plt.ylim([0, 1])
             plt.ylabel('True Positive rate')
             plt.xlabel('Flase positive rate')
             # plt.show()

             # 模型调优

             # 网格搜索验证
             # 第一重搜索  （找出最优的'n_estimators'）
             param_test1 = {'n_estimators': range(1, 30, 2)}
             gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=2,
                                                                      min_samples_leaf=1,
                                                                      max_depth=4),
                                     param_grid=param_test1,
                                     scoring='roc_auc',
                                     cv=2)
             gsearch1.fit(X_train, Y_train)
             # print(gsearch1.best_params_,gsearch1.best_score_)
             # 第二重搜索  （找出最优的'min_samples_split'和 'min_samples_leaf'）
             param_test2 = {'min_samples_split': range(2, 20, 2), 'min_samples_leaf': range(1, 10, 2)}
             gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=1,
                                                                      max_depth=4),
                                     param_grid=param_test2,
                                     scoring='roc_auc',
                                     cv=2
                                     )
             gsearch2.fit(X_train, Y_train)
             # print(gsearch2.best_params_,gsearch2.best_score_)
             # 第三重搜索  （找出最优的'max_depth'）
             param_test3 = {'max_depth': range(1, 10, 1)}
             gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=1,
                                                                      min_samples_split=3,
                                                                      min_samples_leaf=1,
                                                                      ),
                                     param_grid=param_test3,
                                     scoring='roc_auc',
                                     cv=2)
             gsearch3.fit(X_train, Y_train)
             # print(gsearch3.best_params_,gsearch3.best_score_)
             # {'max_depth': 1} 0.5

             # 第四种网格搜索（找出最优的'class_weight'和'criterion'）
             param_test4 = {'class_weight': [None, 'balanced'], 'criterion': ['gini', 'entropy']}
             gsearch4 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=100,
                                                                      min_samples_leaf=1,
                                                                      min_samples_split=3,
                                                                      max_depth=4),
                                     param_grid=param_test4,
                                     scoring='roc_auc',
                                     cv=2)
             gsearch4.fit(X_train, Y_train)
             model_new = gsearch4.best_estimator_.fit(X_train, Y_train)
             y_new = model_new.predict(data_new1)
             list1 = y.columns.tolist()
             str1 = ''.join(list1)

             for item in y_new:
                 if item == 0:
                     str2 = '%s在该店为非优势品类' % str1
                     list2.append(str2)
                 else:
                     str3 = '%s在该店为优势品类' % str1
                     list2.append(str3)
         d1 = pd.DataFrame(list2,index=None)
     return render_template("index.html", output=list2)
app.run(port=8080,debug=True)
if __name__ == '__main__':
    main()





