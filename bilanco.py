from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv


driver = webdriver.Firefox()


url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse=THYAO"
driver.get(url)


time.sleep(2)


button_xpath = '//*[@id="tab1"]/div[1]/div/ul/li[4]'
button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, button_xpath))
)
button.click()


time.sleep(2)


content = driver.find_elements(By.XPATH, '//*[@id="tbodyMTablo"]/tr')  


with open('balance.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    
    writer.writerow(["Başlıklar", "2024 3. çeyrek", "2024 2. çeyrek", "2024 1. çeyrek", "2023 4. çeyrek"])  
    
  
    for row in content:

        columns = row.find_elements(By.TAG_NAME, 'td')
        

        row_data = [col.text.strip() for col in columns]  
        if len(row_data) == 5:  
            writer.writerow(row_data)


driver.quit()
