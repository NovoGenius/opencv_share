from statsmodels.tsa.stattools import adfuller as ADF
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
from matplotlib.pylab import style
style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 导入数据
data = pd.read_excel('/Users/luoliang/Desktop/ADS-各店各单品每日销售汇总_品类趋势.xlsx',index_col='日期')
data_train = data.loc[:'2022-06-23']
# data_test = data.loc['2022-04-01':]
# 数据清洗
std_data = data_train['销额'].std()
avg_data = data_train['销额'].mean()
data_train['z_score'] = (data_train['销额']-avg_data)/std_data
data_clear = data_train.loc[data_train['z_score']<=3]
data_clear = data_clear.drop('z_score',axis=1)
# print(data_clear)
def testA(data):
    m=10
    acf,q,p = sm.tsa.acf(data,nlags=m,qstat=True)
    out = np.c_[range(1,m+1),acf[1:],q,p]
    output = pd.DataFrame(out,columns=['lag','corr','统计量Q值','P值'])
    output = output.set_index('lag')
    print(output)
# testA(data_clear)
#           corr         统计量Q值   P值
# lag
# 1.0   0.715682   3147.470421  0.0
# 2.0   0.596675   5335.576617  0.0
# 3.0   0.569315   7327.939160  0.0
# 4.0   0.555171   9222.845213  0.0
# 5.0   0.564772  11184.175131  0.0
# 6.0   0.634124  13657.175281  0.0
# 7.0   0.697393  16648.764388  0.0
# 8.0   0.594695  18824.496146  0.0
# 9.0   0.509414  20421.220161  0.0
# 10.0  0.491153  21905.760907  0.0

# 模型进行平稳性检验
diff1 = data.diff(1).dropna()
res_adf = ADF(data)[1]
res_adf1 = ADF(diff1)[1]
# diff1.plot()
# data_clear.plot()
# 输出模型的acf和pacf图表
plot_acf(diff1).show()
plot_pacf(diff1).show()
# plt.show()
# 输出p为7，q为10

# 寻找AIC和BIC的信息准则寻找p和q值
# def confirm_q_p(data):
#     AIC = sm.tsa.arma_order_select_ic(data, max_ar=10, max_ma=10, ic='aic')['aic_min_order']
#     BIC = sm.tsa.arma_order_select_ic(data, max_ar=10, max_ma=10, ic='bic')['bic_min_order']
#     print('BIC',BIC)
#     print('AIC',AIC)
# pq = confirm_q_p(diff1)

import pandas as pd
def prediction(data):
    model = sm.tsa.ARIMA(data,order =(7,1,2)).fit()
    pre = model.forecast(7)
    data_pre = pd.DataFrame(round(pre))
    data_clear.plot()
    data_pre.plot()
    plt.show()
    y_pre = data_pre
    print(y_pre)
    # y_ture = data_test.values.sum()
    # print(y_ture)
    # pertent = (y_pre/y_ture)
    # print('预测量精度为', pertent)
# 模型的残差检验
#     resid = model.resid
#     fig = plt.figure(figsize=(12,8))
#     ax = fig.add_subplot(111)
#     fig = qqplot(resid, line='q', ax=ax, fit=True)
#     plt.show()
prediction(data_clear)