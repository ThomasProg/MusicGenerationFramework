# pip install selenium
# sudo apt-get install -y chromium-browser

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Function to scrape using Selenium
def scrape_with_selenium(url):
    # Set up WebDriver (you might need to install a driver like ChromeDriver)
    # Set up ChromeDriver with path
    # chrome_driver_path = 'chromedriver-linux64/chromedriver'  # Specify the path to your chromedriver executable
    # service = Service(chrome_driver_path)
    # driver = webdriver.Chrome(service=service)
    driver = webdriver.Chrome()
    driver.get(url)
    
    try:
        # Wait until the page is fully loaded (adjust timeout as needed)
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.TAG_NAME, "span")))
        
        # Once loaded, get the page source and pass it to BeautifulSoup
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Extract spans with data-id attribute
        spans_with_data_id = soup.find_all('span')
        for span in spans_with_data_id:
            print(span)
            # print(span['data-id'])
            
    finally:
        # Close the WebDriver
        driver.quit()

# Example usage
url = 'https://imslp.org/wiki/Special:IMSLPImageHandler/62144'  # Replace with your target URL
# # url = "https://www.google.com/search?q=f&oq=f&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTINCAEQLhjHARjRAxiABDIGCAIQRRg7MgYIAxBFGDwyBggEEEUYPDIGCAUQRRg8MgYIBhBFGD0yBggHEEUYPNIBBzU4OGowajeoAgCwAgA&sourceid=chrome&ie=UTF-8"

driver = webdriver.Chrome()
driver.get(url)


page_source = driver.page_source
soup = BeautifulSoup(page_source, 'html.parser')
link = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.LINK_TEXT, "I understand")))
link.click()
# spans_with_data_id = soup.find_all('a')

# for span in spans_with_data_id:
#     if (span.text ==  "I understand"):
#         # span["href"]
#         span.click()
#     print(span)

# Wait until the page is fully loaded (adjust timeout as needed)
WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.TAG_NAME, "span")))

# Once loaded, get the page source and pass it to BeautifulSoup
page_source = driver.page_source
soup = BeautifulSoup(page_source, 'html.parser')

# Extract spans with data-id attribute
spans_with_data_id = soup.find_all('span')
for span in spans_with_data_id:
    try:
        dataid = span["data-id"]
        print(dataid)
    except:
        pass

    # print(span['data-id'])
        
# finally:
#     # Close the WebDriver
#     driver.quit()

# scrape_with_selenium(url)