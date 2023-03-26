'''导入pandas的包'''
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
'''读取文件存储的位置'''
data = pd.read_csv(r'/Users/luoliang/Desktop/sales_info1.csv')
'''将日期转化成python能识别的datetime'''
data = data[['门店编码','商品编码','日期','销量','门店库存','事件','日期属性','门店名称','商品名称','印尼大类','印尼中类','印尼小类','印尼细类']]
data['日期'] = pd.to_datetime(data['日期'])
'''将商品编码转化成 字符串格式， python中int，float，object格式 要统一，有时候会出现匹配错的问题，查看字段格式使用data.info()'''
data['商品编码'] = data['商品编码'].astype('object')
'''查看清洗前的数据'''
print('清洗前的数据销量为',data['销量'].sum())
i = 0
'''这里循环20次，可以根据数据量进行灵活替换，通常建议不小于3次'''
while i <= 20:
    '''数据透视'''
    '''1,以门店和编码聚合计算每个产品的均值'''
    data_avg = data.pivot_table(index=['门店编码','商品编码'],values='销量',aggfunc='mean')
    '''2,以门店和编码聚合计算每个产品的标准差'''
    data_std = data.pivot_table(index=['门店编码','商品编码'],values='销量',aggfunc='std')
    '''将avg和std 拼接回原数据'''
    data = pd.merge(data,data_std,left_on=['门店编码','商品编码'],right_on=['门店编码','商品编码'],how='left',suffixes=('','_std'))
    data = pd.merge(data,data_avg,left_on=['门店编码','商品编码'],right_on=['门店编码','商品编码'],how='left',suffixes=('','_avg'))
    '''计算每天的标准化系数'''
    data['z_score'] =( data['销量'] - data['销量_avg']) / data['销量_std']
    ''' 如果该sku 只有一条数据 那么 其标准差为NULL值,进行null值替换为0'''
    data.fillna(0,inplace=True)
    '''筛选出3倍标准差内'''
    data_nor = data.loc[(data['z_score'] <= 3) & (data['z_score'] >= -3)]
    '''筛选出3倍标准差外'''
    data_disnor = data.loc[(data['z_score'] > 3) | (data['z_score'] < -3)]
    '''计算异常sku编码的个数'''
    num = data_disnor['商品编码'].count()
    '''通过异常sku的个数去判断 是否有必要进行数据的替换'''
    if num != 0 :
        '''删除异常sku的销量，用均值进行替代'''
        del data_disnor['销量']
        '''python的merge函数相当于 excel中的 vlookup'''   '''3倍标准差外用该sku在该门店的均值替换'''
        data_disnor = pd.merge(data_disnor,data_avg,left_on=['门店编码','商品编码'],right_on=['门店编码','商品编码'],how='left')
        '''再把3倍标准外 替换后的数据集 拼接回原来的数据集 再进行异常值的判断'''
        data = pd.concat([data_nor,data_disnor],axis=0)
        '''将上一次循环的标准差以及均值删除，防止数据干扰'''
        del data['销量_std']
        del data['销量_avg']
        del data['z_score']
        '''循环后的data进行保存，再当作第二次循环的data进行数据清洗'''
        data = data
    else:
        data = data_nor
    i += 1
print('清洗后的数据销量为',data['销量'].sum())
data['门店编码'] = data['门店编码'].astype('str')
print(data.info())
'''数据保存，文件路径直接  打开一个excel的属性，复制路径，用/分割 ，文件路径前面要加 r '''
print(data)
# data.to_excel(r'/Users/luoliang/Desktop/clear_data.xlsx')





