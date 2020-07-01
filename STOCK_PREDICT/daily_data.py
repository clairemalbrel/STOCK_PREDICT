##########################################################################
#     Stock Predict dail dataset
##########################################################################

import datetime as dt
import json
import requests
import requests
from datetime import datetime
import time
import pandas as pd
import math


##########################################################################
#     Reddit scrap
##########################################################################

text = []
name = []
today = dt.datetime.combine(dt.date.today(), dt.datetime.min.time())
response = requests.get('https://www.reddit.com/r/worldnews/top.json?limit=40', headers = {'User-agent': 'test'})
tops = json.loads(response.text)['data']['children']
i = 1
for top in tops:
    if dt.datetime.fromtimestamp(top['data']['created']) >= today:
        text.append(top['data']['title'])
        name.append(f'Top{i}')
        i = i + 1

#full = clean("".join(str(text)))
data = pd.DataFrame(text).T
data.columns = name
data = data.iloc[:,0:25]
data['Date'] = datetime.date(today)

##########################################################################
#     Yahoo Finance
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

# sort and calculate change
new =  new.sort_values('Date')
new['change'] = new['Open'].pct_change()
new['change'] = new['change'].shift(-1)

def categorical(x):
    if x >= 0:
        x = 1
    if x < 0:
        x = 0
    if math.isnan(x):
        x = 'today'
    return x

new['target'] = new['change'].apply(categorical)

##########################################################################
#     Merge and add to csv
##########################################################################

y0 = data.merge(new, how='left')
y0.head()
print(y0)
#y0.to_csv('stock_predict_daily.csv')

y0.to_csv('data/daily_data.csv', mode='a', header=False)

