##########################################################################
#     Reddit Scrap
##########################################################################
import datetime as dt
import json
import requests
import pandas as pd
from SentencePolarity.sentiment import Sentiment



def reddit():
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
  data = pd.DataFrame(text).T
  data.columns = name
  data = data.iloc[:,0:25]
  return data
df=reddit()

sentiment = Sentiment()
sentence_polarity_infomap = {}
#filepath = 'stocknews/Combined_News_DJIA2.csv'
combined_stock_data = df
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
print(combined_stock_data.head())

