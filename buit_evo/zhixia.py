from selenium import webdriver
import time
import csv
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

def seach(words):
    driver.find_element_by_xpath('//*[@id="app"]/div/main/div[2]/div[2]/div/div[1]/p[2]').click()
    time.sleep(2)
    login = '17304413561'
    pswd = '7702675ff'
    driver.find_element_by_xpath('//*[@id="app"]/div/main/div[2]/div[2]/div/form[2]/div[1]/div/div[1]/input').send_keys(login)
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="app"]/div/main/div[2]/div[2]/div/form[2]/div[2]/div/div/input').send_keys(pswd)
    time.sleep(2)
    '''手动滑块验证码'''
    driver.find_element_by_xpath('//*[@id="app"]/div/main/div[2]/div[2]/div/form[2]/div[3]/div/button').click()
    time.sleep(18)
    """打开侧边框"""
    # driver.find_element_by_xpath('//*[@id="hamburger-container"]').click()
    # time.sleep(2)
    '''切换国家'''
    driver.find_element_by_xpath('//*[@id="app"]/div/div[2]/div[1]/div[1]/div[2]/div[2]').click()
    time.sleep(3)
    '''选择印尼'''
    f = driver.find_element_by_xpath('//*[@id="app"]/div/div[2]/div[1]/div[1]/div[2]/div[2]')
    f.find_element_by_xpath("//*[text()='印度尼西亚']").click()
    time.sleep(3)


    '''热搜词与标签词 只能一个个进行，不能两者同时进行  Ctrl+/ 是快速注释代码'''

    """                                      标签词分析                         """
    # driver.find_element_by_xpath('//div[@class = "el-scrollbar__view"]/ul/div[10]').click()
    # time.sleep(2)
    '''1.1   热销标签词   只能三选一'''
    # driver.find_element_by_xpath('//*[@id="app"]/div/div[1]/div[2]/div[1]/div/ul/div[10]/li/ul/div[1]/a/li/span').click()
    '''1.2   飙升标签词   只能三选一'''
    # driver.find_element_by_xpath('//*[@id="app"]/div/div[1]/div[2]/div[1]/div/ul/div[10]/li/ul/div[2]/a/li/span').click()
    '''1.3   热销新词     只能三选一'''
    # driver.find_element_by_xpath('//*[@id="app"]/div/div[1]/div[2]/div[1]/div/ul/div[10]/li/ul/div[3]/a/li/span').click()



    driver.find_element(By.XPATH,'//div[@class = "el-scrollbar__view"]/ul/div[11]')
    """                                      热搜词分析                         """
    driver.find_element_by_xpath('//div[@class = "el-scrollbar__view"]/ul/div[11]').click()
    time.sleep(2)
    '''2.1  热销词    只能三选一'''
    driver.find_element_by_xpath('//*[@id="app"]/div/div[1]/div[2]/div[1]/div/ul/div[11]/li/ul/div[1]/a/li/span').click()
    '''2.2  飙升热词  只能三选一'''
    # driver.find_element_by_xpath('//*[@id="app"]/div/div[1]/div[2]/div[1]/div/ul/div[11]/li/ul/div[2]/a/li/span').click()
    '''2.3  热销新词  只能三选一'''
    # driver.find_element_by_xpath('//*[@id="app"]/div/div[1]/div[2]/div[1]/div/ul/div[11]/li/ul/div[3]/a/li/span').click()
    time.sleep(3)



    '''标签词专用     只能二选一 '''
    # s = '//*[@id="app"]/div/div[2]/section/div/div[2]/div[1]/div/div/input'

    '''热搜词专用     只能二选一'''
    s = '//*[@id="app"]/div/div[2]/section/div/div[2]/div[1]/div[1]/div/div/input'



    '''清空输入框'''
    driver.find_element_by_xpath(s).clear()
    time.sleep(2)
    '''输入关键字'''
    driver.find_element_by_xpath(s).send_keys(words)
    time.sleep(2)
    '''点击搜索'''
    driver.find_element_by_xpath('/html/body/div[4]/div[2]/div[1]/ul/li[1]').click()
    time.sleep(2)

def get_information():
    '''获取热搜词'''
    s = driver.find_elements_by_xpath('//div[@class="el-table__fixed-body-wrapper"]/table/tbody/tr[@class="el-table__row"]')
    for ls in s:
        title  = ls.find_element_by_xpath('.//td[1]/div/p').text
        msg1 = {'''热度词为：%s'''%(title)}
        print(msg1)
        '''保存数据至桌面'''
        with open(r'C:\Users\Administrator\Desktop\美妆1.csv', mode='a', newline='',encoding='utf-8') as filecsv:
            f = csv.writer(filecsv, delimiter=',')
            f.writerow([title])

    '''获取数据'''

    p = driver.find_elements_by_xpath('//*[@id="app"]/div/div[2]/section/div/div[3]/div[1]/div[3]/table/tbody/tr[@class="el-table__row"]')
    for lp in p :
        SKU    = lp.find_element_by_xpath('.//td[2]/div/span').text
        pcs_30 = lp.find_element_by_xpath('.//td[3]/div/span').text
        pcs_7  = lp.find_element_by_xpath('.//td[4]/div/span').text
        pcs_1  = lp.find_element_by_xpath('.//td[5]/div/span').text
        price_avg = lp.find_element_by_xpath('.//td[7]/div/span').text
        msg2 = {'''
        有效SKU数为：%s
        最新30天销量：%s
        最近7天销量为：%s
        最近一天销量为：%s
        均价为：%s
        '''%(SKU,pcs_30,pcs_7,pcs_1,price_avg)}
        # print(msg2)
        '''保存数据至桌面'''
        with open(r'C:\Users\Administrator\Desktop\美妆.csv', mode='a', newline='',encoding='utf-8') as filecsv:
            f = csv.writer(filecsv, delimiter=',')
            f.writerow([SKU,pcs_30,pcs_7,pcs_1,price_avg])

def scroll(driver):
    driver.execute_script(""" 
        (function () { 
            var y = document.body.scrollTop; 
            var step = 100; 
            window.scroll(0, y); 
            function f() { 
                if (y < document.body.scrollHeight) { 
                    y += step; 
                    window.scroll(0, y); 
                    setTimeout(f, 50); 
                }
                else { 
                    window.scroll(0, y); 
                    document.title += "scroll-done"; 
                } 
            } 
            setTimeout(f, 1000); 
        })(); 
        """)
def next_paper():
    paper=1
    while paper <= 2:
        time.sleep(2)
        driver.find_element_by_xpath('//*[@id="app"]/div/div[2]/section/div/div[3]/div[2]/div/span[3]/div/input').send_keys(Keys.CONTROL,"a")
        time.sleep(2)
        driver.find_element_by_xpath('//*[@id="app"]/div/div[2]/section/div/div[3]/div[2]/div/span[3]/div/input').send_keys(paper,Keys.ENTER)
        time.sleep(2)
        paper +=1
        get_information()
def main():
    seach(words)
    next_paper()
if __name__ == '__main__':
    words = input('请输入需要查询得关键词：')
    driver = webdriver.Chrome()
    driver.get('https://shopee.mobduos.com/#/home')
    driver.maximize_window()
    main()
