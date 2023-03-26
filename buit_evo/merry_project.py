import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["font.family"] = 'Arial Unicode MS'
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings('ignore')
data = pd.read_excel(r'/Users/luoliang/Desktop/日常工作文件/圣诞节备货/ads_总部采购订单明细.xlsx')
data_group1 = data.groupby('商品编码')['采购交期'].mean()
data_group2 = data.groupby('商品编码')['采购交期'].std()
# group_mean = data_group.采购交期.mean()
# group_std = data_group.采购交期.std()
data_mean = data_group1.to_frame()
data_std = data_group2.to_frame()
data_base = pd.merge(data_mean,data_std,how='left',on='商品编码' )
data_base.reset_index(inplace=True)
data_base.rename(columns={'商品编码':'商品编码','采购交期_x':'mean','采购交期_y':'std'},inplace=True)
'''引入产品重要程度'''
data_2 = pd.read_excel(r'/Users/luoliang/Desktop/日常工作文件/圣诞节备货/商品明细-3.xlsx',sheet_name=1)
data_3 = data_2[['商品编码','商品类别_印尼']]

data_base = pd.merge(data_base,data_3,how='left',on='商品编码')
'''A类产品安全备货3系数'''
'''B类产品安全备货1.8系数'''
'''C类产品安全备货1.2系数'''
da = data_base.loc[data_base['商品类别_印尼'] == 'A类']
da['corr'] = da['std'] * 2.8
db = data_base.loc[data_base['商品类别_印尼'] == 'B类']
db['corr'] = db['std'] * 1.8
dc = data_base.loc[data_base['商品类别_印尼'] == 'C类']
dc['corr'] = dc['std'] * 1.2
dd = data_base.loc[(data_base['商品类别_印尼'] == 'T-A类') | (data_base['商品类别_印尼'] == 'T-B类')]
dd['corr'] = dd['std'] * 1



data_total = pd.concat([da,db,dc,dd],axis=0)
print(data_total)

# data_total.to_excel(r'/Users/luoliang/Desktop/日常工作文件/圣诞节备货/SKU安全备货天数.xlsx')
'''画分布图'''
# sb.histplot(data_std)
sb.histplot(data_mean)
plt.show()
