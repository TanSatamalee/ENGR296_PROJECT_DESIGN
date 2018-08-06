import requests
import urllib
import re
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import secrets

pd.set_option('display.max_columns', None)

# Returns pandas array of reviews from requested steam url.
def get_reviews(url):
	# Requests for the page and strips html from it.
	page = requests.get(url)
	soup = BeautifulSoup(page.content, 'html.parser')

	# Begins crawling through the page for first 10 items.
	review_blocks = soup.find_all('div', class_='apphub_Card')

	# Initialize data array.
	df = pd.DataFrame(columns=['username', 'helpful votes', 'funny votes', 'overall rating', 'hours played', 'date posted', 'review'])
	review_num = 0
	for b in review_blocks:
		df.loc[review_num] = _extract_block(b)
		review_num += 1

	# Continues the crawling by requesting for the rest of the pages.
	for i in range(2, 1000):
		new_url = _get_request_url(url, i)

		# Creates header and asks site for content.
		hdr = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0'}
		req = urllib.request.Request(new_url, headers=hdr)
		response = urllib.request.urlopen(req)

		# Terminates if the response does not have status code of 200.
		if response.getcode() != 200:
			break

		soup = BeautifulSoup(response, 'html.parser')
		review_blocks = soup.find_all('div', class_='apphub_Card')
		
		for b in review_blocks:
			df.loc[review_num] = _extract_block(b)
			review_num += 1

	print("Finished recording " + str(review_num) + " reviews.")
	return df

# Stores pandas array into filename specified.
def store_reviews(df, filename):
	df.to_csv(filename)
	print("Successfully stored!")

# Extracts all information from a html block for a single review and returns one numpy array.
def _extract_block(b):
	# Extracts username using profile url to get alphanumeric characters (easier to work with).
	name = b.find('div', class_='apphub_friend_block_container').find('a')['href'].split("/")[-2]
	# Extracts the votes for helpful and funny of the review.
	helpful_count = re.findall('\d+', b.find('div', class_='found_helpful').getText())
	# Extracts overall opinion.
	rating = b.find('div', class_='title').getText()
	# Extracts total number of hours played.
	hours = b.find('div', class_='hours').getText().split()[0]
	# Extracts date review was posted.
	date = b.find('div', class_='date_posted').getText()[8:]
	# Extracts review content.
	review = " ".join(b.find('div', class_='apphub_CardTextContent').getText().split()[(len(date.split()) + 1):])

	# Stores found data (username, helpful votes, funny votes, overall rating, hours played, date posted, review)
	data = np.array([name, helpful_count[0], helpful_count[1], rating, hours, date, review])
	return data

# Returns the request url for the nth page of reviews.
def _get_request_url(url, n):
	n = str(n)
	parse_url = url.split('/')
	req_url = '/'.join(parse_url[:5]) + '/homecontent/?userreviewsoffset=' + n + '0&p=' + n + '&workshopitemspage=' + n + '&readytouseitemspage=' + n + '&mtxitemspage=' + n + '&itemspage=' + n + '&screenshotspage=' + n + '&videospage=' + n + '&artpage=' + n + '&allguidepage=' + n + '&webguidepage=' + n + '&integratedguidepage=' + n + '&discussionspage=' + n + '&numperpage=10&browsefilter=toprated&browsefilter=toprated&appid=' + parse_url[4] + '&appHubSubSection=10&appHubSubSection=10&l=english&filterLanguage=default&searchText=&forceanon=1'
	return req_url

store_reviews(get_reviews(secrets.dota_url), secrets.dota_file)
print(pd.read_csv(secrets.dota_file))
