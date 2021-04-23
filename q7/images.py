from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time 
#Opens up web driver and goes to Google Images
driver = webdriver.Chrome(r'C:\Users\Shril Mody\Desktop\chromedriver.exe')
driver.get('https://www.google.ca/imghp?hl=en&tab=ri&authuser=0&ogbl')

box = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
box.send_keys('snowy owl')
box.send_keys(Keys.ENTER)

last_height = driver.execute_script('return document.body.scrollHeight')
while True:
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(2)
    new_height = driver.execute_script('return document.body.scrollHeight')
    try:
        driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
        time.sleep(2)
    except:
        pass
    if new_height == last_height:
        break
    last_height = new_height

# //*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img
path = r"D:\imp docs\study\Semesters\sem 6\ML\Assignments\ES654-assignmnet-3-shrilboss\Images\Snowy Owl\\"
for i in range(1, 100):
    try:
        driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').screenshot(path+str(i)+".png")
    except:
        pass