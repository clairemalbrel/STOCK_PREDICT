##########################################################################
#     Reddit Scrap
##########################################################################

import datetime as dt
import json
import requests

text = []
today = dt.datetime.combine(dt.date.today(), dt.datetime.min.time())
response = requests.get('https://www.reddit.com/r/worldnews/top.json?limit=25', headers = {'User-agent': 'test'})
tops = json.loads(response.text)['data']['children']
for top in tops:
  if dt.datetime.fromtimestamp(top['data']['created']) >= today:
    text.append(top['data']['title'])
  else:
    text.append(top['data']['created'])

full = clean("".join(str(text)))
new_df = pd.DataFrame(data, columns = ['Date', 'cleaned'])
new_df
