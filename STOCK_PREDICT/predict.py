##########################################################################
#     Data to predict
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
import requests
from datetime import datetime
import time
import datetime as dt
import json
import requests
import requests
from datetime import datetime
import time
import pandas as pd
import math
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
from SentencePolarity.sentiment import Sentiment
from sklearn.preprocessing import RobustScaler, StandardScaler
from STOCK_PREDICT.data import Data
import joblib
import requests
from datetime import datetime
import time
from termcolor import colored
import warnings


##########################################################################
#     Predict
##########################################################################

directory = str(Path.home()) + '/code/stock_market/STOCK_PREDICT/'
if platform.system() == 'Windows':
  directory = directory.replace('\\' ,'/')
directory2 = str(Path.home()) + '/code/stock_market/front/'


class Predict:

  def __init__(self):
    self.sentiment = pd.read_csv(f'{directory}data2/sentiment_scores.csv')

  def reddit(self):

    # Get timestamp of today
    ts = int(time.time())

    # yaho API
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-historical-data"

    querystring = {"frequency":"1d","filter":"history","period1":"1592179200","period2":str(ts),"symbol":"%5EDJI"}

    headers = {
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com",
        'x-rapidapi-key': "9f58e76de3msh0050a18aab7452fp15c352jsn04ff99cb8ff8"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)
    re = response.json()

    # Bring response to dataframe format
    major_liste = []

    for i in re['prices']:
        liste =[datetime.fromtimestamp(i['date']),
                i['open'],
                i['high'],
                i['low'],
                i['close'],
                i['volume'],
                i['adjclose']]
        major_liste.append(liste)

    new = pd.DataFrame(major_liste)
    new.columns = ['Date','Open','High','Low','Close','Volume','Adj Close']
    new['Date'] = pd.to_datetime(new['Date'])
    new['Date'] = new['Date'].dt.date

    # change
    change = new
    change =  change.sort_values('Date')
    change['target'] = change['Open'].pct_change()
    change['target'] = change['target'].apply(categorical)
    change['target'] = change['target'].shift(-1)
    change = change[['Date','target']]
    return new, change

  def merge(self):
    new, change = self.reddit()
    sentiment = self.sentiment
    sentiment['Date'] = pd.to_datetime(sentiment['Date'])
    sentiment['Date']  = sentiment['Date'].dt.date
    check = sentiment.merge(new)
    final = check[['Date','Subjectivity','Objectivity','Positive','Negative','Neutral','Open','High','Low','Close','Volume','Adj Close']]
    check = check.iloc[:,27:]
    check = check[['Subjectivity','Objectivity','Positive','Negative','Neutral','Open','High','Low','Close','Volume','Adj Close']]
    return check, final

  def scaling(self, X_train):
    check, final = self.merge()
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    scaled = scaler.transform(check)
    return scaled

  def pred(self, pipeline, X_train):
    new, change = self.reddit()
    check, final = self.merge()
    scaled = self.scaling(X_train)
    pipeline = pipeline
    res = pipeline.predict(scaled)
    if res > 0:
      final['prediction'] = 'Rise'
    if res < 0:
      final['prediction'] = 'Fall'
    final = final.merge(change)
    return final

if __name__ == "__main__":
  warnings.simplefilter(action='ignore', category=FutureWarning)
  print(colored("############  Prepare the data   ############", "red"))
  predict = Predict()
  reddit = predict.reddit()
  check = predict.merge()
  data = Data()
  df, X_train, X_test, y_train, y_test, df_train, df_test = data.clean_df()
  scaled = predict.scaling(X_train)
  print(scaled.shape)
  print(colored("############  Load pretrained model ############", "blue"))
  pipeline = joblib.load('model.joblib')
  final = predict.pred(pipeline, X_train)
  print(final['prediction'])
  print(colored("############   Save Prediction   ############", "green"))
  # save to backend
  final.to_csv('data2/final.csv', mode='a', header=False)
  # save to frontend
  final.to_csv(f'{directory2}data2/final.csv', mode='a', header=False)

  print('Prediction saved to csv')
