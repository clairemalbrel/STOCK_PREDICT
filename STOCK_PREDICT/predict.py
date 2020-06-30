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

##########################################################################
#     Code
##########################################################################
directory = str(Path.home()) + '/code/stock_market/STOCK_PREDICT/'
if platform.system() == 'Windows':
  directory = directory.replace('\\' ,'/')

class Predict:

  def __init__(self):
    self.daily = pd.read_csv(f'{directory}data/daily_data.csv')

  def data(self):
    return self.daily

if __name__ == "__main__":
  predict = Predict()
  daily = predict.data()
  print(daily.shape)



