import pandas as pd
import numpy as np
import math
# 输入新店总面积
areas_new = float(input('请输入新店面积:'))
distance = float(input('请输入新店距离仓库的距离（公里数）:'))
new_store_money = float(input('请输入新店租金/月:'))
# 新店每平米的租金
every_areas_money = new_store_money/areas_new
# 加载各个场景的占地面积
data_evo = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/场景面积sps基础数据.xlsx')
data_evo_min = data_evo.loc[data_evo['是否必须上']=='Y']
# 最小的场景面积
data_evo_min1 = data_evo_min.场景面积.sum()
# print(data_evo_min1)
# 最大的场景面积
data_evo_max = data_evo.场景面积.sum()
# 可再增加的场景
data_evo_sd = int(data_evo_max - data_evo_min1)

# 收银台面积
msgCheckStand = 17

# 加载出各个距离的物流费用
data_trs = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/物流费用sps基础数据.xlsx')
# 计算仓库距离对于不同车次的费用核算
data_big_car = data_trs.loc[data_trs['车辆类型'] == '大车']
data_mid_car = data_trs.loc[data_trs['车辆类型'] == '中车']
data_sma_car = data_trs.loc[data_trs['车辆类型'] == '小车']
# 100公里内的物流费用
data_big_pur_100 = distance * data_big_car['100公里内']
data_mid_pur_100 = distance * data_mid_car['100公里内']
data_sma_pur_100 = distance * data_sma_car['100公里内']
# 100公里外的费用
data_big_pur = distance * data_big_car['100公里外']
data_mid_pur = distance * data_mid_car['100公里外']
data_sma_pur = distance * data_sma_car['100公里外']
# 不同车大小与实际荷载量
big_max_load = data_big_car['车辆容积']
mid_max_load = data_mid_car['车辆容积']
sma_max_load = data_sma_car['车辆容积']

# =====================================  店铺与仓库距离之间，周转天数的分类   =====================================
if distance <= 100:
    sales_day = 6
elif distance <= 500:
    sales_day = 8
elif distance <= 1000:
    sales_day = 10
else:
    sales_day = 15
print('====================       标准店的铺面积      ===================')
print('该新店标准仓库备货天数：',sales_day)
# 导入商品规划明细（基础数据集  so important important！！！！！！！）
data_base = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/印尼规划sku取数明细-2.xlsx')
standrd_sku = data_base.商品编码.count()
# print(standrd_sku)
# =====================================       计算标准营业面积以及仓库面积      =====================================
data_paimian = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/SKU排面量基础数据.xlsx')
data_concet = pd.merge(data_base,data_paimian,how='left',left_on='商品编码',right_on='商品编码')
data_concet = data_concet.fillna(0)
# 背柜陈列米数测算
data_concet_behind = data_concet.loc[data_concet['陈列货架类型']=='背柜']
data_toys = data_concet_behind.loc[data_concet_behind['商品大类']=='玩具']
data_toys = data_toys.copy()
data_toys['display_mi'] = data_toys.长度mm*0.001 * data_toys.合理排面量 / 4/1
# 沐浴用品
data_muyu = data_concet_behind.loc[data_concet_behind['商品小类']=='沐浴用品']
data_muyu = data_muyu.copy()
data_muyu['display_mi']= data_muyu.长度mm*0.001 * data_muyu.合理排面量 / 4/1
#香氛蜡烛
data_xiangfeng = data_concet_behind.loc[data_concet_behind['商品中类']=='香氛蜡烛']
data_xiangfeng = data_xiangfeng.copy()
data_xiangfeng['display_mi'] = data_xiangfeng.长度mm*0.001 * data_xiangfeng.合理排面量 / 4/1

data1 = pd.concat([data_xiangfeng,data_muyu,data_toys],axis=0)

sample = data1['商品编码']
data_concet_behind = data_concet_behind.copy()
data_concet_behind['jug'] = data_concet_behind['商品编码'].isin(sample)
data3 = data_concet_behind.loc[data_concet_behind['jug'] == False]
data3 = data3.copy()
data3['display_mi'] = data3.长度mm*0.001 * data3.合理排面量 / 5/1

# 背柜的陈列米数
data_concet_behind1 = pd.concat([data3,data1],axis=0)

behind_long = data_concet_behind1.display_mi.sum()
# 背柜面积测算
standrd_beigui_areas = round(behind_long)
# print('背柜',standrd_beigui_areas)

# 标准店铺的陈列量成本
data_display_money = data_concet.合理排面量 * data_concet.深度量 * data_concet.成本价
print('标准店铺的陈列量成本',int(data_display_money.sum()))

# 中岛陈列米数测算
data_concet_mid = data_concet.loc[data_concet['陈列货架类型']=='中岛']

# 玩具
data_toys1 = data_concet_mid.loc[data_concet_mid['商品大类']=='玩具']
data_toys1 = data_toys1.copy()
data_toys1['display_mi'] = data_toys1.长度mm*0.001 * data_toys1.合理排面量 / 4/1
# 沐浴用品
data_muyu1 = data_concet_mid.loc[data_concet_mid['商品小类']=='沐浴用品']
data_muyu1 = data_muyu1.copy()
data_muyu1['display_mi']= data_muyu1.长度mm*0.001 * data_muyu1.合理排面量 / 4/1
#香氛蜡烛
data_xiangfeng1 = data_concet_mid.loc[data_concet_mid['商品中类']=='香氛蜡烛']
data_xiangfeng1 = data_xiangfeng1.copy()
data_xiangfeng1['display_mi'] = data_xiangfeng1.长度mm*0.001 * data_xiangfeng1.合理排面量 / 4/1

data4 = pd.concat([data_xiangfeng1,data_muyu1,data_toys1],axis=0)

sample = data4['商品编码']
data_concet_mid = data_concet_mid.copy()
data_concet_mid['jug'] = data_concet_mid['商品编码'].isin(sample)
data5 = data_concet_mid.loc[data_concet_mid['jug'] == False]
data5 = data5.copy()
data5['display_mi'] = data5.长度mm*0.001 * data5.合理排面量 / 5/1

data_concet_mid1 = pd.concat([data4,data5],axis=0)

mid_long = data_concet_mid1.display_mi.sum()
# data_concet_mid.to_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/xxx.xlsx')
# 中岛面积测算
standrd_zhondao_areas = round(mid_long/2/3*11)
# print('中岛',standrd_zhondao_areas)

# 特殊货架测算
data_concet_spc = data_concet.loc[data_concet['陈列货架类型']=='彩妆柜']
data_concet_spc = data_concet_spc.copy()
spc_long = sum(data_concet_spc.长度mm*0.001 * data_concet_spc.合理排面量 / 4/0.29)

# 特殊货架sku 陈列米数明细
data_concet_spc['display_mi'] = data_concet_spc.长度mm*0.001 * data_concet_spc.合理排面量/ 4
# 特殊货架面积测算
standrd_caizhuanggui_areas = round(spc_long/2/12) * 9
# print(standrd_caizhuanggui_areas)
# 指甲油台
spc_zhijia_areas = 6

# 各大类陈列米数明细计算
data_sku_display_mi = pd.concat([data_concet_spc,data_concet_behind1,data_concet_mid1],axis=0)
# data_sku_display_mi.to_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/xxx.xlsx')
# 用于品类规划测算陈列米数
df1 = data_sku_display_mi.pivot_table(index=['商品大类','商品中类','商品小类'],values='display_mi',aggfunc='sum')
# df1.to_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/galaxy_sku1.xlsx')


# print('货架使用面积：',int(standrd_beigui_areas+standrd_caizhuanggui_areas+standrd_zhondao_areas+spc_zhijia_areas))
print('营业面积的货架米数',int(behind_long+mid_long+spc_long))
# 标准门店营业面积由各个货架的面积以及收银台、泡面墙组成，由于店铺的不规整，逻辑取数带来的误差，允许误差范围10%，增加一个修正系数
standrd_sales_areas =round ((standrd_zhondao_areas+standrd_beigui_areas\
                      +standrd_caizhuanggui_areas\
                      +spc_zhijia_areas+data_evo_min1+msgCheckStand)*1.1)
print("标准营业面积为:",standrd_sales_areas,"坪")

# =====================================   计算仓库标准面积    =====================================
# 店铺日均销量（日均销量为3000pcs）
psd = data_base.店均日均销量.sum()
# 存放一天的货对仓库的需求面积
pro_height = data_base['店均日均销量'] * data_base['长度mm']*0.001 * data_base['宽度mm']*0.001 * data_base['高度mm']*0.001
# 计算店铺位置相对合理备货天数的门店仓库占地面积
stock_areas = pro_height.sum()/0.8*1.3*sales_day

# 员工休息区
setRestLengthScale = 6
# 将货架个数转化成货架占地面积以及增加员工休息区，由于店铺的不规整，逻辑取数带来的误差，允许误差范围10%，增加一个修正系数
standrd_stock_areas = round(stock_areas*1.1 + setRestLengthScale*1.1)

# =======================================      计算标准店铺营业面积       =========================================
# 营业面积和仓库面积求和
standrd_areas  = int(round(standrd_sales_areas + standrd_stock_areas))
print("标准仓库面积为:",standrd_stock_areas,"坪")
print("标准店铺面积为:", standrd_areas, "坪（标准店：使用标准排面量，使用标准备货天数，使用必要上的场景）")





# print('====================       最大的饱和店铺      ===================')
# =====================================       计算最大营业面积以及仓库面积      =====================================
data_paimian = pd.read_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/SKU排面量基础数据.xlsx')
data_concet = pd.merge(data_base,data_paimian,how='left',left_on='商品编码',right_on='商品编码')
data_concet = data_concet.fillna(0)
# 背柜陈列米数测算
data_concet_behind = data_concet.loc[data_concet['陈列货架类型']=='背柜']
behind_long_m = sum(data_concet_behind['长度mm']*0.001 * data_concet_behind['最大排面量'] / 6/1)
# 背柜面积测算
standrd_beigui_areas_m = round(behind_long_m)
# print('背柜',standrd_beigui_areas)

# 最大店铺的陈列量成本
data_display_money_m = data_concet.最大排面量 * data_concet.深度量 * data_concet.成本价
# print('最大店铺的陈列量成本',int(data_display_money_m.sum()))

# 中岛陈列米数测算
data_concet_mid = data_concet.loc[data_concet['陈列货架类型']=='中岛']
mid_long_m = sum(data_concet_mid['长度mm']*0.001 * data_concet_mid['最大排面量'] / 5/1)
# print(mid_long)
# 中岛面积测算
standrd_zhondao_areas_m = round(mid_long_m/2/3*11)
# print('中岛',standrd_zhondao_areas)

# 特殊货架测算
data_concet_spc = data_concet.loc[data_concet['陈列货架类型']=='彩妆柜']
spc_long_m = sum(data_concet_spc['长度mm']*0.001 * data_concet_spc['最大排面量'] / 5/0.29)

# 特殊货架面积测算
standrd_caizhuanggui_areas = round(spc_long/2/12) * 9
# print(standrd_caizhuanggui_areas)
# 指甲油台
spc_zhijia_areas = 6


# 最大门店营业面积由各个货架的面积以及收银台、泡面墙组成，由于店铺的不规整，逻辑取数带来的误差，允许误差范围10%，增加一个修正系数
standrd_sales_areas_m =round ((standrd_zhondao_areas_m +standrd_beigui_areas_m \
                      +standrd_caizhuanggui_areas\
                      +spc_zhijia_areas+data_evo_max+msgCheckStand)*1.1)
# print("最大营业面积为:",standrd_sales_areas_m ,"坪")

# =====================================   计算最大仓库标准面积    =====================================
# 店铺日均销量（日均销量为3000pcs）
# psd = data_base.店均日均销量.sum()
# 存放一天的货对仓库的需求体积
pro_height = data_base['店均日均销量'] * data_base['长度mm']*0.001 * data_base['宽度mm']*0.001 * data_base['高度mm']*0.001
# 计算店铺位置相对合理备货天数的门店仓库占地面积（30天为极限最大值）
stock_areas_m = pro_height.sum()/0.8*1.3*30

# 员工休息区
setRestLengthScale = 6
# 将货架个数转化成货架占地面积以及增加员工休息区，由于店铺的不规整，逻辑取数带来的误差，允许误差范围10%，增加一个修正系数
standrd_stock_areas_m = round(stock_areas_m*1.1 + setRestLengthScale*1.1)

# =======================================      计算最大店铺营业面积       =========================================
# 营业面积和仓库面积求和
# standrd_areas_m  = int(round(standrd_sales_areas_m + standrd_stock_areas_m))
# print("最大仓库面积为:",standrd_stock_areas_m,"坪")
# print("最大店铺面积为:", standrd_areas_m, "坪（最大店：使用最大排面量，使用最大备货天数(30天)，场景全上）")




# ===================================       满足开设小店的条件       ===========================================
# 新小店的面积最小为1200
new_stock_bult  = int(1200)

# ====================         不满足小店的条件时，应该增大排面量还是增大门店仓库的承载量     =========================
# 一天的销量对仓库占地面积
if areas_new > standrd_areas:
    print('～～～～～～～～～～～～～～     由于该新店面积过大 对新店规划如下，请各个部门注意查收    ～～～～～～～～～～～～')
    # print('预测新店的销售额为:',int(psd*30*13))
    data_base1 = data_base[['商品编码','商品大类','商品中类','商品小类','商品细类','商品类别','品牌名称']]
    data_all = pd.merge(data_base,data_paimian,on='商品编码',how='left',suffixes=('_x','_y'))
    data_all['陈列米数'] = (data_all.合理排面量 * data_all.长度mm * 0.001) /5
    data_pvoted = data_all.pivot_table(index='商品大类',values='陈列米数',aggfunc='sum')
    print(data_pvoted)


    data_all.to_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/大店sku上架明细.xlsx')
    # 大店判断增大仓库面积是否有必要T
    # 一天的销量对门店仓库的面积占用
    stock_areas = pro_height.sum() / 0.8 * 1.3
    # 条件判断是否需要增大门店仓库面积
    list1 = [0]
    for i in range(1, 31 - sales_day):
        # jd仓费用计算
        jd_stock = int((30 / sales_day * math.ceil(psd * sales_day / big_max_load) * data_big_pur) +
                       (psd * i * 30 / (sales_day + i) * 30 * 0.0036))
        # 门店费用计算
        store_stock = int((30 / (sales_day + i) * math.ceil(psd * (sales_day + i) / big_max_load) * data_big_pur) +
                          (stock_areas * i * every_areas_money))

        if jd_stock > store_stock:
            list1.append(i)
    add_stock_areas = int(stock_areas * max(list1))

    # 判断盈余面积如何进行分配
    more_areas = areas_new - standrd_areas
    if more_areas >= new_stock_bult:
        print('建设将盈余面积做小店：',more_areas)
    elif more_areas >= data_evo_sd:
        more_areas1 = more_areas - data_evo_sd
        print('==================  场景端  ================== ')
        print('建议在原计划场景的基础上增加场景为：',int(data_evo_sd))
        if more_areas1 >= add_stock_areas:
            more_areas2 = more_areas1 - add_stock_areas
            print('==================  设计端  ================== ')
            print('原计划门店仓库面积设定为：', standrd_stock_areas)
            print('建议门店在原有基础上再增加仓库面积为：', int(add_stock_areas))
            print('==================  计调端  ================== ')
            print('原计划门店仓库备货天数为：', sales_day)
            print('建议门店仓库端在原有基础上再增加的备货天数为：', max(list1))
            print('================  品类规划端  ================= ')
            print('原计划仓库货架货架米数为：', int(pro_height.sum() * sales_day))
            print('建议在原计划的基础上再增大货架米数为：', int(pro_height.sum() * max(list1)))
            print('=================   陈列端   ================= ')
            print('建议增加产品排面量的面积为：',int(more_areas2))
        else:
            print('==================  计调端  ================== ')
            print('原计划门店仓库备货天数为：', sales_day)
            add_days = math.ceil(more_areas1/(pro_height.sum() / 0.8 * 1.3))
            print('建议门店在原有基础上增加的备货天数:',add_days)
            print('==================  设计端  ================== ')
            print('原计划门店仓库面积设定为：', standrd_stock_areas)
            print('建议门店在原有基础上增加仓库面积为:',int(more_areas1))
            print('================  品类规划端  ================= ')
            print('原计划仓库货架货架米数为：', int(pro_height.sum() * sales_day))
            print('建议在原计划的基础上再增大货架米数为：', int(pro_height.sum() * add_days))
    else:
        print('==================  场景端  ================== ')
        print('建议增加的场景面积为：',int(more_areas))

else:
    avg_sku_sales = standrd_sales_areas / standrd_sku
    avg_sku_stock = standrd_stock_areas / standrd_sku
    sku1 = round(areas_new / (avg_sku_sales + avg_sku_stock))
    sales_need = round(sku1 * avg_sku_sales)
    stock_need = round(sku1 * avg_sku_stock)
    msg2 = {''' 建议实际的营业面积为:%s
                建议实际的仓库面积为:%s
                建议上架SKU数为:%d
                ''' % (sales_need, stock_need, sku1)}
    print(msg2)
    # 以上得出店铺的实际营业面积以及仓库面积，以及该店的上架sku数量
    # 核心sku必上
    sku_imp = data_concet.loc[data_concet['必上sku判断'] == '必上sku']
    # 未来到店的新品必上
    sku_new_pro = data_concet.loc[data_concet['必上sku判断'] == '新品']
    # 合并核心产出产品以及新品
    sku_must = pd.concat([sku_imp, sku_new_pro], axis=0)
    # 统计必上sku的数量
    sku_min = sku_must.商品编码.count()
    # 统计所有的sku数量
    sku_max = data_base.商品编码.count()
    # 选择完必上sku后，对剩余的产品进行rank选择
    sku_more = sku1 - sku_min

    # 筛选出非必上sku的明细
    data_not_imp = data_concet.loc[data_concet['必上sku判断']=='非必上sku']
    # 再通过剩余sku的数量，筛选出需要上架的sku
    sku_more_need = data_not_imp.loc[(data_not_imp['非必上sku排名'] <= sku_more) & (data_not_imp['非必上sku排名'] > 0)]
    # 最终小店的上架明细
    sku_planning = pd.concat([sku_must,sku_more_need],axis=0)
    # 小店日均销量预测
    psd_small = sku_planning.店均日均销量.sum()
    # 小店月度销售额预测（13为件单价）
    pre_money = int(psd_small*30*13)
    print('新店月度销售额预测值为：',pre_money)
    sku_planning.to_excel(r'/Users/luoliang/Desktop/项目开发文件/SPS项目/sps开发文件/小店上架sku明细.xlsx')
    # 计算各个品类大类上架的sku以及其的陈列米数
    # ==============================================    大类米数规划     =================================================
    sku_planning['mishu'] = (sku_planning.合理排面量 * sku_planning.长度mm * 0.001) / 5
    data_pivoted1 = sku_planning.pivot_table(index=['商品大类'], values='mishu', aggfunc='sum')
    print('大类米数规划')
    print(data_pivoted1)
    

