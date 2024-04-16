import os
import time
import random
import requests
from bs4 import BeautifulSoup

def get_html_from_chrome():
	
	import applescript
    
	# appleScript to get the HTML of the current tab in Google Chrome
	script = '''
	tell application "Google Chrome"
		set currentTab to active tab of front window
		execute currentTab javascript "document.documentElement.outerHTML"
	end tell
	'''

	# execute the AppleScript and capture the output
	try:
		result = applescript.run(script)
		if result.out:
			return result.out  # return the HTML content
		else:
			return "Failed to capture HTML: " + result.err
	except Exception as e:
		return str(e)



# get all of the dat files
def extract_dat_links(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    a_tags = soup.find_all('a', href=True)
    dat_links = [a['href'] for a in a_tags if a['href'].endswith('.dat')]
    return dat_links


# downloading all of the dat files that we need
def download_dat_files(urls, save_folder='Example Data'):

	# create the download folder
	os.makedirs(save_folder, exist_ok=True)

	# iterate through each URL to download the file
	for url in urls:
		
		time.sleep(0.5)
		
		# randomly wait
		if random.random() < 0.1:
			time.sleep(1)
		
		filename = url.split('/')[-1]
		save_path = os.path.join(save_folder, filename)
		
		# get the data
		response = requests.get(url, stream=True)
		
		# check that the download was ok
		if response.status_code == 200:
			
			# write the output
			with open(save_path, 'wb') as f:
				for chunk in response.iter_content(chunk_size=8192):
					f.write(chunk)
			print(f'Downloaded {filename} to {save_path}')
		else:
			print(f'Failed to download {url}')
			
	print('Download process completed.')


# deciding whether we should get the data links from the website or from the local file
USE_WEBSITE = False
if USE_WEBSITE:

	html_content = get_html_from_chrome()
	dat_links = extract_dat_links(html_content)

	# go through each of the links and add the right prefix to them
	dat_links = ["https://m-selig.ae.illinois.edu/ads/" + d_link for d_link in dat_links]

else:

	# using the local file
	dat_links = open("./uiuc_data_links.txt", 'r').readlines()
	dat_links = [dl.strip() for dl in dat_links]


# download all of the data links
download_dat_files(dat_links)
