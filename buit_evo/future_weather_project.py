import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings("ignore")
from selenium.webdriver.common.by import By
import time
city = input('请输入要查询未来天气的城市：')
driver = webdriver.Chrome()
driver.get('https://tianqi.2345.com')
'''输入查询城市'''
driver.find_element_by_xpath('//*[@id="js_searchInput"]').send_keys(city)
driver.find_element_by_xpath('//*[@id="js_searchBtn"]').click()
'''爬取数据'''
def get_information():
    date1=[]
    weather1=[]
    temp1 =[]
    wind1 = []
    p = driver.find_elements_by_xpath('/html/body/div[7]/div[2]/div[2]/ul/li')
    for lp in p:
        date = lp.find_element_by_tag_name('span').text
        weather = lp.find_element_by_class_name('how-day').text
        temp = lp.find_element_by_class_name('tem-show').text
        wind = driver.find_element_by_class_name('home-day').text
        date1.append(date)
        weather1.append(weather)
        temp1.append(temp)
        wind1.append(wind)
    data_date = pd.DataFrame(date1)
    data_weather = pd.DataFrame(weather1)
    data_temp = pd.DataFrame(temp1)
    data_wind = pd.DataFrame(wind1)
    data = pd.concat([data_date,data_weather,data_temp,data_wind],axis=1)
    name = ['date','weather','temp','wind']
    data.columns = name
    '''数据转化处理'''
    data['date'] = data['date'].apply(lambda x:x.split(' ')[0])
    data['min_temp'] = data['temp'].apply(lambda x:x.split('~')[0])
    data['max_temp'] = data['temp'].apply(lambda x:x.split('~')[1])
    data['max_temp'] = data['max_temp'].apply(lambda x: x.split('°')[0])
    data['wind'] = data['wind'].str.extract('(\d+)').fillna(data['wind'])
    del data['temp']

    data['date'] = data['date'].str.replace('/','-')
    data['date'] = '2022-' + data['date']
    data['date'] = pd.to_datetime(data['date'])

    print(data)
get_information()

