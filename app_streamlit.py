##########################################################################
#     Libraries
##########################################################################
import joblib
import pandas as pd
import pytz
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import time
from STOCK_PREDICT.data import Data



##########################################################################
#     Application
##########################################################################

@st.cache
def read_data():
  data = Data()
  df, X_train, X_test, y_train, y_test = data.clean_df()
  return df, X_train, X_test, y_train, y_test


df, X_train, X_test, y_train, y_test, df_train, df_test = data.clean_df()
st.title("Stock Market Prediction using Sentiment Analysis")
today = dt.datetime.combine(dt.date.today(), dt.datetime.min.time())
today = datetime.date(today)
st.header(today)

@st.cache

st.dataframe(df_test.head())


data = Data()
df, X_train, X_test, y_train, y_test = data.clean_df()
pipeline = joblib.load('model.joblib')
print("loaded model")

X_to_pred = pd.DataFrame(X_test.iloc[0]).T
res = pipeline.predict(X_to_pred)
if res > 0:
  st.markdown('The Stock price will rise')
if res < 0:
  st.markdown('The Stock price will decrease')

#if __name__ == "__main__":
