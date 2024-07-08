# pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
import os

# Function to get the page content
def get_page_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to retrieve page: {response.status_code}")
        return None

# # Function to parse the page and extract PDF links
# def extract_pdf_links(content):
#     soup = BeautifulSoup(content, 'html.parser')
#     pdf_links = []
#     for link in soup.find_all('a', href=True):
#         href = link['href']
#         if href.endswith('.pdf'):
#             pdf_links.append(href)
#     return pdf_links

# Function to parse the page and extract PDF links
def extract_pdf(content):
    soup = BeautifulSoup(content, 'html.parser')
    pdf_links = []

    for b in soup.find_all('b'):
        print(b)

    for link in soup.find_all('span'):
        if ('data-id' in link):
            href = link['data-id']
            if href.endswith('.pdf'):
                pdf_links.append(href)
    return pdf_links



# Function to parse the page and extract PDF links
def extract_links(content):
    soup = BeautifulSoup(content, 'html.parser')
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        links.append(href)
    return links


# Function to download a file from a URL
def download_file(url, directory='downloads'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    local_filename = os.path.join(directory, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# Example usage
# imslp_url = 'https://imslp.org/wiki/Category:C_major'
# imslp_url = 'https://imslp.org/wiki/10_Esquisses%2C_Op.82_(Mo%C3%B3r%2C_Emanuel)'
imslp_url = "https://imslp.org/wiki/Special:ImagefromIndex/62144/wc13"
page_content = get_page_content(imslp_url)
if page_content:
    pdf_links = extract_pdf(page_content)

    print(pdf_links)

    # for pdf_link in pdf_links:
    #     # Construct full URL for relative links
    #     if pdf_link.startswith('/'):
    #         pdf_link = f'https://imslp.org{pdf_link}'
    #     print(f"Downloading {pdf_link}")
    #     download_file(pdf_link)









