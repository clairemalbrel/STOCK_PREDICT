
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
import getpass
from pathlib import Path
import os
import platform
from STOCK_PREDICT.utils import *

directory = str(Path.home()) + '/code/stock_market/STOCK_PREDICT/'
if platform.system() == 'Windows':
	directory = directory.replace('\\' ,'/')


class Data:

	def __init__(self):
		#path2=f'/Users/{USERNAME}/code/{USERNAME}/STOCK_PREDICT/data/Combined_News_DJIA.csv'
		#username = getpass.getuser()
		#path1=f'/Users/{USERNAME}/code/{USERNAME}/STOCK_PREDICT/data/datasets-129-792900-upload_DJIA_table.csv'
		#df_news = pd.read_csv(f'{directory}/data/Combined_News_DJIA.csv')

		#df_djia = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets-129-792900-upload_DJIA_table.csv')
		#df_news = os.path.join(os.path.dirname(__file__), '..', 'data', 'Combined_News_DJIA.csv')
		#df_djia = pd.read_csv(df_djia)
		#df_news = pd.read_csv(df_news)

		self.df_djia = pd.read_csv(f'{directory}data/datasets-129-792900-upload_DJIA_table.csv')
		self.df_news = pd.read_csv(f'{directory}data/Combined_News_DJIA.csv')


	def clean_df(self):
		#merge dataset
		df_djia = self.df_djia
		df_news = self.df_news
		df_news['Date'] = pd.to_datetime(df_news['Date'])
		df_djia['Date'] = pd.to_datetime(df_djia['Date'])
		df = df_news.merge(df_djia)

		# percentage change
		df['change'] = df['Open'].pct_change()
		# remove first row
		df['change'] = df['change'].shift(-1)
		#
		df['target'] = df['change'].apply(categorical)
		# Group news
		cols = df.columns[2:]
		df['combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
		#
		df['cleaned'] = df['combined'].apply(clean)
		# Sentiment scores
		df['polarity'] = df['cleaned'].apply(polarity)
		df['subjectivity'] = df['cleaned'].apply(subjectivity)

        # Vader features
		df['compound'] = df['cleaned'].apply(compound)
		df['negative'] = df['cleaned'].apply(negative)
		df['neutrale'] = df['cleaned'].apply(neutrale)
		df['positive'] = df['cleaned'].apply(positive)
		return df


	if __name__ == "__main__":
		df = Data().clean_df()
		print(df.shape)

