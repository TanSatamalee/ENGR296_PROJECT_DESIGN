import pandas as pd
import numpy as np
import json
import string
import re
import nltk
import time
import operator
import math
from sklearn import svm
from matplotlib import pyplot as plt

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
	s = re.sub('[^A-Za-z ]', '', s.lower()).split()
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


# Given a dataframe with a column named 'reveiw', removes reviews that do not meet the min and max word count requirements.
def df_wordcount(df, min=0, max=500):
	# Checks to see if there is a 'review' column in the dataframe.
	if 'review' not in df.columns:
		print("Dataframe did not contain a 'review' column for punctuation removal.")
		return df

	need_drop = []
	for index, row in df.iterrows():
		word_count = len(row['review'].split())
		if word_count < min or word_count > max:
			need_drop.append(index)

	return df.drop(need_drop).reset_index(drop=True)


# Returns a list of possible values the ngrams could be.
def ngram_dict(df, n):
	# Checks to see if n is sufficient.
	if n < 1:
		print("n is too small!")
		return None

	# Creating a 2D array of ngrams for each user review.
	ngram = dict()
	for i in range(0, df.shape[0]):
		s = df.loc[i]['review'].split()
		for j in range(0, len(s)-n+1):
			word = " ".join(s[j:j+n])
			if word in ngram:
				ngram[word] += 1
			else:
				ngram[word] = 1

	while n > 1:
		ngram.update(ngram_dict(df, n - 1))
		n -= 1
	
	# Getting top 5 ngrams.
	return dict(sorted(ngram.items(), key=operator.itemgetter(1), reverse=True))


# Given a dataframe with a column named 'review' and an int 'n', returns a new dataframe with columns of ngram counts of the reviews.
def ngrams(df, n, ngram):
	# Creating a 2D array of ngrams for each user review.
	ngram_df = pd.DataFrame(columns=ngram.keys())
	for i in range(0, df.shape[0]):
		s = df.loc[i]['review'].split()
		temp = pd.DataFrame(0, index=[0], columns=ngram.keys())
		for j in range(0, len(s)-n+1):
			word = " ".join(s[j:j+n])
			if word in ngram.keys():
				temp[word] += 1

		ngram_df = ngram_df.append(temp)

	while n > 1:
		ngram_df.append(df, n - 1, ngram)
		n -= 1
	
	return ngram_df


# Given a dataframe with a column named 'review' and an int 'n', returns a new dataframe with columns of ngram entropy of the reviews.
def ngrams_entro(df, n, ngram):
	# Creating a 2D array of ngrams for each user review.
	ngram_df = pd.DataFrame(columns=ngram.keys())
	for i in range(0, df.shape[0]):
		s = df.loc[i]['review'].split()
		temp = pd.DataFrame(0, index=[0], columns=ngram.keys())
		for j in range(0, len(s)-n+1):
			word = " ".join(s[j:j+n])
			if word in ngram.keys():
				temp[word] += 1

		ngram_df = ngram_df.append(temp)

	ngram_sum = ngram_df.sum(axis=1).replace(0, 1)
	ngram_df = ngram_df.div(ngram_sum, axis=0)
	ngram_df = ngram_df.applymap(lambda x: -1 * x * math.log10(x) if x > 0 else 0)
	return ngram_df


# -------------------------------------------------------------------


# Find the k most/least important features by creating an SVM for the target from the dataframe.
def find_feats(df, target, feats, k):
	svmer = svm.SVC(kernel='linear')
	svmer.fit(df, target)
	imp, names = zip(*sorted(zip(svmer.coef_[0], feats))[:k] + sorted(zip(svmer.coef_[0], feats))[-k:])
	plt.barh(range(len(names)), imp, align='center')
	plt.yticks(range(len(names)), names)
	plt.show()

	corr, ngram = zip(*sorted(zip(svmer.coef_[0], feats)))
	return dict(zip(ngram, corr))
 
# -------------------------------------------------------------------

'''
t0 = time.time()
d = {'review': ['probably one of the best games i have ever played. omg this is sooooooo good. i would buy this for everyone i know!!', 'I didnt really like this game', 'hellyessssseriy', '', 'duhhhhhhhhhhhh duh duhh hi'], 'recommend': [1, 0, 1, 0, 0]}
data = pd.DataFrame(d)
#data = read_csv("data/steam.csv", ['id', 'review', 'recommend', 'helpful'])
print(data.head(5))
data2 = df_rm_dup(data.head(30000))
print(data2.head(5))
data3 = df_rm_punc(data2)
print(data3.head(5))
data4 = df_rm_stopw(data3, ['easy', 'learn', 'games'])
print(data4.head(5))
data5 = df_stem(data4)
print(data5.head(5))
data6 = df_wordcount(data5, 2, 100)
print(data6.head(5))
data_dict = ngram_dict(data6, 2)
print(len(data_dict))
data_dict2 = {key: value for key, value in data_dict.items() if value > 0}
print(len(data_dict2))
data7 = ngrams_entro(data6, 2, data_dict2)
print(data7.shape)
print(data7.head(5))

t5 = time.time()
print("Overall Time: " + str(t5 - t0))
'''
