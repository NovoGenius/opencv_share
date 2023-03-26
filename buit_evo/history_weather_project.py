import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings("ignore")
from selenium.webdriver.common.by import By
import time
driver = webdriver.Chrome()
driver.get('https://tianqi.2345.com/inter_history/asia/jakarta.htm')
'''建立一个list用于装数据'''
data= []
def seach():
    '''选择年份'''
    driver.find_element(By.XPATH,'//*[@id="js_yearVal"]').click()
    time.sleep(0.5)
    driver.find_element(By.XPATH,'/html/body/div[7]/div[2]/div[1]/div[1]/div[1]/div[3]/div[1]/div/ul/li[8]/a').click()
    time.sleep(0.5)
    '''选择月份'''
    driver.find_element(By.XPATH,'//*[@id="js_monthVal"]').click()
    time.sleep(0.5)
    driver.find_element(By.XPATH,'/html/body/div[7]/div[2]/div[1]/div[1]/div[1]/div[3]/div[2]/div/ul/li[1]/a').click()
    time.sleep(1)
    get_information()
    '''实现下一页的翻页'''
    for i in range(1,3):
        next_paper()
    treanfor()
def get_information():
    elment = driver.find_element_by_class_name('box-mod-tb')
    tr_elment = elment.find_element_by_tag_name('tbody').find_elements_by_tag_name('tr')
    for tr in tr_elment:
        td_elment = tr.find_elements_by_tag_name('td')
        lst=[]
        for td in td_elment:
            lst.append(td.text)
        data.append(lst)
def next_paper():
    driver.find_element_by_xpath('//*[@id="js_nextMonth"]').click()
    time.sleep(1)
    get_information()
def treanfor():
    '''将爬取的数据进行dataframe的转化'''
    data1 = pd.DataFrame(data,columns=['week_date','max_temp1','min_temp1','weather','wind'])
    data1.dropna(axis=0,inplace=True)
    '''日期转化'''
    data1['date'] = data1.week_date.apply(lambda x:x.split(' ')[1])
    data1['week'] = data1.week_date.apply(lambda x:x.split(' ')[0])
    del data1['week_date']
    '''最高温度转化'''
    data1['max_temp'] = data1.max_temp1.apply(lambda x:x.split('°')[0])
    del data1['max_temp1']
    '''最低问题转化'''
    data1['min_temp'] = data1.min_temp1.apply(lambda x:x.split('°')[0])
    del data1['min_temp1']
    '''正则表达式提取风级'''
    data1['wind'] = data1['wind'].str.extract('(\d+)').fillna(data1['wind'])
    print(data1)
    '''保存数据到文件，可直接覆盖之前数据'''
    data1.to_csv(r'/Users/luoliang/Desktop/项目开发文件/新销量预测文件/XGboost/天气数据.csv')

if __name__ == '__main__':
    seach()


