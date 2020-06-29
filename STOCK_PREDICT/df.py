
USERNAME='Jean-Phi-ben'
# path1 et path2 à rajouter au notebook avant de charger get_data avec son path individuel
# path1=f'/home/jp/code/STOCK_PREDICT/data/upload_DJIA_table.csv'
# path2=f'/home/jp/code/STOCK_PREDICT/data/Combined_News_DJIA.csv'

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
import getpass
from pathlib import Path
import os
import platform


def get_data():
    #path2=f'/Users/{USERNAME}/code/{USERNAME}/STOCK_PREDICT/data/Combined_News_DJIA.csv'
    #username = getpass.getuser()
    #path1=f'/Users/{USERNAME}/code/{USERNAME}/STOCK_PREDICT/data/datasets-129-792900-upload_DJIA_table.csv'
    #df_news = pd.read_csv(f'{directory}/data/Combined_News_DJIA.csv')

    #df_djia = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets-129-792900-upload_DJIA_table.csv')
    #df_news = os.path.join(os.path.dirname(__file__), '..', 'data', 'Combined_News_DJIA.csv')
    #df_djia = pd.read_csv(df_djia)
    #df_news = pd.read_csv(df_news)
    directory = str(Path.home()) + '/code/stock_market/STOCK_PREDICT/'
    if platform.system() == 'Windows':
        directory = directory.replace('\\' ,'/')


    df_djia = pd.read_csv(f'{directory}data/datasets-129-792900-upload_DJIA_table.csv')
    df_news = pd.read_csv(f'{directory}data/Combined_News_DJIA.csv')
def get_data(path1, path2, nrows):
    # path1=f'/home/jp/code/STOCK_PREDICT/data/upload_DJIA_table.csv'
    # path2=f'/home/jp/code/STOCK_PREDICT/data/Combined_News_DJIA.csv'
    df_djia = pd.read_csv(path1, nrows=nrows)
    df_news = pd.read_csv(path2,nrows=nrows)
    return df_djia, df_news

def merge_df(df_djia, df_news):
    #merge dataset
    df_djia, df_news = get_data()
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    df_djia['Date'] = pd.to_datetime(df_djia['Date'])
    df = df_news.merge(df_djia)
    return df

def get_features(df):
    # get percentage change
    df['change'] = df['Open'].pct_change()
    # remove first row
    df['change'] = df['change'].shift(-1)
    return df

def categorical(x):
    if x > 0:
        x = 1
    else:
        x = 0
    return x

def get_cat_data(df):
    df['target'] = df['change'].apply(categorical)
    return df

def group_news(df):
    # Combine the top 25 daily news into 1 column
    cols = df.columns[2:]
    df['combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    return df

def clean(text):
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for punctuation in punctuation:
        review1 = text.replace(punctuation, ' ') # Remove Punctuation

    lowercased = text.lower() # Lower Case
    without_b=text.replace("b ","")
    without_b=text.replace("b'","")
    without_b=text.replace('b"',"")
    tokenized = word_tokenize(without_b) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    lemma=WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
    return " ".join(lemmatized)

def get_clean():
    df_djia, df_news = get_data()
    df=merge_df()
    df=get_features(df)
    df=get_cat_data(df)
    df=group_news(df)
    df['cleaned'] = df['combined'].apply(clean)

if __name__ == '__main__':
    df_djia, df_news = get_data()
    #df_djia = get_data(github_username, nrows=100)
    #df_news = get_data(github_username, nrows=100)
    df=merge_df()
    df=get_features(df)
    df=get_cat_data(df)
    df=group_news(df)
    df['cleaned'] = df['combined'].apply(clean)
    print('df all cleaned and ready')
    print(df.shape)
