from datetime import datetime
import joblib
import pandas as pd
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
from STOCK_PREDICT.data import Data



data = Data()
df, X_train, X_test, y_train, y_test = data.clean_df()
pipeline = joblib.load('model.joblib')
print("loaded model")
X_to_pred = pd.DataFrame(X_test.iloc[0]).T
res = pipeline.predict(X_to_pred)
if res > 0:
  print('The Stock price will rise')
if res < 0:
  print('The Stock price will decrease')

