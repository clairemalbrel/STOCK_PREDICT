##########################################################################
#     Reddit Scrap
##########################################################################

import datetime as dt
import json
import requests
import pandas as pd


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

  #full = clean("".join(str(text)))
  data = pd.DataFrame(text).T
  data.columns = name
  data = data.iloc[:,0:25]
  return data
