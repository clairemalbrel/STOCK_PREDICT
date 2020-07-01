##########################################################################
#     Reddit Scrap
##########################################################################
import json
import datetime as dt
from datetime import datetime
import requests
import pandas as pd
from SentencePolarity.sentiment import Sentiment



##########################################################################
#     Reddit scrap
##########################################################################
text = []
name = []
today = dt.datetime.combine(dt.date.today(), dt.datetime.min.time())
response = requests.get('https://www.reddit.com/r/worldnews/top.json?limit=100', headers = {'User-agent': 'test'})
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
date = datetime.date(today)
data.insert(loc=0, column='Date', value=date)

print(len(data))
##########################################################################
#    get sentiments
##########################################################################

#data = pd.read_csv("data/stock_predict_daily.csv", encoding= 'unicode_escape')
sentiment = Sentiment()
sentence_polarity_infomap = {}
#filepath = 'stocknews/Combined_News_DJIA2.csv'
combined_stock_data = data
combined_stock_data['Para'] = combined_stock_data['Top1']
for x in range(2, 26):
  combined_stock_data['Para'] += combined_stock_data['Top'+str(x)]

for index, sentence in combined_stock_data['Para'].iteritems():
  sentence_polarity_infomap = sentiment.analyze([sentence])
  print("polarity" + str(sentence_polarity_infomap))
  if not (sentence_polarity_infomap == {}):
            combined_stock_data.at[index, 'Subjectivity'] = sentence_polarity_infomap['subjective']
            combined_stock_data.at[index, 'Objectivity'] = sentence_polarity_infomap['objective']
            combined_stock_data.at[index, 'Positive'] = sentence_polarity_infomap['positive']
            combined_stock_data.at[index, 'Neutral'] = sentence_polarity_infomap['neutral']
            combined_stock_data.at[index, 'Negative'] = sentence_polarity_infomap['negative']



##########################################################################
#     Generate csv
##########################################################################
combined_stock_data.to_csv('data2/sentiment_scores.csv', mode='a', header=False)
