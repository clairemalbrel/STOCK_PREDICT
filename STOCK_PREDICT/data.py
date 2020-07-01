
##########################################################################
#     Data
##########################################################################

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


##########################################################################
#     Code
##########################################################################


directory = str(Path.home()) + '/code/stock_market/STOCK_PREDICT/'
if platform.system() == 'Windows':
	directory = directory.replace('\\' ,'/')


class Data:

	def __init__(self):


		#df_news = pd.read_csv(f'{directory}/data/Combined_News_DJIA.csv')

		#df_djia = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets-129-792900-upload_DJIA_table.csv')
		#df_news = os.path.join(os.path.dirname(__file__), '..', 'data', 'Combined_News_DJIA.csv')
		#df_djia = pd.read_csv(df_djia)
		#df_news = pd.read_csv(df_news)

    #Rest of the team:
		#self.df_djia = pd.read_csv(f'{directory}data/datasets-129-792900-upload_DJIA_table.csv')
		#self.df_news = pd.read_csv(f'{directory}data/combined_stock_data.csv')

		#CLAIRE ONLY --> METTRE HASTAG SI VOUS ETES PAS CLAIRE
<<<<<<< HEAD
		USERNAME='clairemalbrel'
		path2=f'/Users/{USERNAME}/code/{USERNAME}/STOCK_PREDICT/data/combined_stock_data.csv'
		path1=f'/Users/{USERNAME}/code/{USERNAME}/STOCK_PREDICT/data/datasets-129-792900-upload_DJIA_table.csv'
		self.df_djia = pd.read_csv(path1)
		self.df_news = pd.read_csv(path2)
=======
		#USERNAME='clairemalbrel'
		#path2=f'/Users/{USERNAME}/code/{USERNAME}/STOCK_PREDICT/data/combined_stock_data.csv'
		#username = getpass.getuser()
		#path1=f'/Users/{USERNAME}/code/{USERNAME}/STOCK_PREDICT/data/datasets-129-792900-upload_DJIA_table.csv'
		#self.df_djia = pd.read_csv(path1)
		#self.df_news = pd.read_csv(path2)
>>>>>>> master


	def clean_df(self):
		#merge dataset
		df_djia = self.df_djia
		df_news = self.df_news
		df_news['Date'] = pd.to_datetime(df_news['Date'])
		df_djia['Date'] = pd.to_datetime(df_djia['Date'])
		df_news['combined'] = df_news[df_news.columns[3:28]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

		# clean news
		df_news['cleaned'] = df_news['combined'].apply(clean)
		df = df_news[['Date', 'Label', 'Subjectivity', 'Objectivity', 'Positive','Negative', 'Neutral', 'cleaned']].merge(df_djia)
		# percentage change to categorical
		df['change'] = df['Open'].pct_change()
		df['change'] = df['change'].shift(-1)
		df['target'] = df['change'].apply(categorical)
		df.index = df.index.sort_values()

		# missing values
		df = df.iloc[:-1]
		# Replace missing values with mean
		nan_list = ['Subjectivity', 'Objectivity', 'Positive', 'Negative', 'Neutral']
		for col in nan_list:
			df[col] = df[col].fillna(df[col].mean())

		# Train test split
		y = df['target']
		# Define eligible Y variable
		X = df.drop('Label', axis = 1)
		X = X.drop('target', axis = 1)
		X = X.drop('Date', axis = 1)
		X = X.drop('change', axis = 1)
		X = X.drop('cleaned', axis = 1)
		train_size = int(len(X.index) * 0.7)

		# train/test split
		df_train = df.loc[:train_size, :]
		df_test = df.loc[train_size:, :]
		X_train, X_test = X.loc[0:train_size, :], X.loc[train_size: len(X.index), :]
		y_train, y_test = y[0:train_size+1], y.loc[train_size: len(X.index)]

		return df, X_train, X_test, y_train, y_test, df_train, df_test

if __name__ == "__main__":
    # Get and clean data
    data = Data()
    df, X_train, X_test, y_train, y_test, df_train, df_test = data.clean_df()
    print(df.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape, df_train.shape, df_test.shape)
