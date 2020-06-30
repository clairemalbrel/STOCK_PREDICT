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

##########################################################################
#     Reddit
##########################################################################



def date_time(x):
    x = datetime.date(x)
    return x


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
new['Date']=new['Date'].apply(date_time)
new['Date']=pd.to_datetime(new['Date'])
print(new)

##########################################################################
#     Merge
##########################################################################

directory = str(Path.home()) + '/code/stock_market/STOCK_PREDICT/'
if platform.system() == 'Windows':
  directory = directory.replace('\\' ,'/')

sentiment = pd.read_csv(f'{directory}data/20200630_sentiment.csv')
sentiment['Date'] = pd.to_datetime(sentiment['Date'])

check = sentiment.merge(new)
print(check.shape)
print(sentiment)
print(check['Open'])
check = check.iloc[:,27:]
print(check.columns)
check[['Subjectivity','Objectivity','Positive','Negative','Neutral','Open','High','Low','Close','Volume','Adj Close']]
print(check.columns)
##########################################################################
#     Scaler
##########################################################################
data = Data()
df, X_train, X_test, y_train, y_test, df_train, df_test = data.clean_df()

scaler = StandardScaler()
scaler = scaler.fit(X_train)
check = scaler.transform(check)
print(check)

##########################################################################
#     Load model and predict
##########################################################################
pipeline = joblib.load('model.joblib')
print("loaded model")

res = pipeline.predict(check)
if res > 0:
  print('The Stock price will rise')
if res < 0:
  print('The Stock price will decrease')
