import pandas as pd
import numpy as np
import json
import string
import re
import nltk
import time

# Given a CSV filename and its column names, this returns a pandas array of the contents.
def read_csv(filename, columns=None):
	if columns is None:
		return pd.read_csv(filename, header=None)
	df = pd.read_csv(filename)
	df.columns = columns
	return df


# Given a JSON filename, this returns a pandas array of the contents.
def read_json(filename):
	data = []
	for f in open(filename, 'r'):
		data.append(json.loads(f))
	return pd.DataFrame.from_dict(data, orient='columns')


# -------------------------------------------------------------------


# Given a dataframe with a column named 'review', removes all duplicates (including original) and returns same dataframe.
def df_rm_dup(df):
	# Checks to see if there is a 'review' column in the dataframe.
	if 'review' not in df.columns:
		print("Dataframe did not contain a 'review' column for punctuation removal.")
		return df

	return df.drop_duplicates('review', keep=False).reset_index(drop=True)


# Given a dataframe with a column named 'review', removes the punctuation and returns same dataframe.
def df_rm_punc(df):
	# Checks to see if there is a 'review' column in the dataframe.
	if 'review' not in df.columns:
		print("Dataframe did not contain a 'review' column for punctuation removal.")
		return df

	table = {ord(char): None for char in string.punctuation}
	df['review'] = df['review'].apply(lambda x: x.translate(table))
	return df


# Given one string, removes the selected words and returns lowercase string.
def word_strip(s, stop_words):
	s = re.sub('[^A-Za-z0-9 ]', '', s.lower()).split()
	return " ".join(list(filter(lambda x: x not in stop_words, s)))


# Given a dataframe with a column named 'review' and a list of words, removes the words and returns same dataframe.
def df_rm_stopw(df, stop_words):
	# Checks to see if there is a 'review' column in the dataframe.
	if 'review' not in df.columns:
		print("Dataframe did not contain a 'review' column for punctuation removal.")
		return df

	df['review'] = df['review'].apply(word_strip, args=[stop_words])
	return df


# Given one string, returns the string with words stemmed.
def word_stem(s):
	stemmer = nltk.stem.PorterStemmer()
	s = s.split()
	stem_s = []
	for i in s:
		stem_s.append(stemmer.stem(i))
	return " ".join(stem_s)


# Given a dataframe with a column named 'review', stems all words and returns the dataframe.
def df_stem(df):
	# Checks to see if there is a 'review' column in the dataframe.
	if 'review' not in df.columns:
		print("Dataframe did not contain a 'review' column for punctuation removal.")
		return df

	df['review'] = df['review'].apply(word_stem)
	return df


# Given a dataframe with a column named 'review' and an int 'n', returns a new dataframe with columns of ngram counts of the reviews.
def ngrams(df, n):
	# Checks to see if n is sufficient.
	if n < 1:
		print("n is too small!")
		return None

	# Creating a 2D array of ngrams for each user review.
	ngram = []
	for i in range(0, df.shape[0]):
		s = df.loc[i]['review'].split()
		temp = []
		for j in range(0, len(s)-n+1):
			temp.append(" ".join(s[j:j+n]))
		ngram.append(temp)
	
	# Finding unique entires of the ngrams for the new dataframe columns.
	ungram = set(ngram[0])
	for i in ngram[1:]:
		ungram.update(i)
	num_columns = len(ungram)
	ungram = list(ungram)
	
	# Creating new dataframe to contain the ngrams.
	ngram_df = pd.DataFrame(columns=ungram)
	n = 0
	for i in ngram:
		arr = np.zeros(num_columns)
		for j in range(0, num_columns):
			if ungram[j] in i:
				arr[j] += 1
		ngram_df.loc[n] = arr
		n += 1
	
	return ngram_df


# -------------------------------------------------------------------

t0 = time.time()
#d = {'review': ['probably one of the best games i have ever played. omg this is sooooooo good. i would buy this for everyone i know!!', 'I didnt really like this game', 'hellyessssseriy', '', 'duhhhhhhhhhhhh duh duhh hi'], 'helpful': [1, 0, 1, 0, 0]}
#data = pd.DataFrame(d)
data = read_csv("data/steam.csv", ['id', 'review', 'recommend', 'helpful'])
print(data.head(5))
data2 = df_rm_dup(data.head(30000))
print(data2.head(5))
data3 = df_rm_punc(data2)
print(data3.head(5))
data4 = df_rm_stopw(data3, ['easy', 'learn', 'games'])
print(data4.head(5))
data5 = df_stem(data4)
print(data5.head(5))
ngram2s = ngrams(data5, 2)
print(ngram2s.shape)
print(ngram2s.head(5))

t5 = time.time()
print("Overall Time: " + str(t5 - t0))
