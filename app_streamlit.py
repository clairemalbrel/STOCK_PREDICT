##########################################################################
#     Libraries
##########################################################################
import joblib
import pandas as pd
import numpy as np
import pytz
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
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
  df, X_train, X_test, y_train, y_test, df_train, df_test = data.clean_df()
  return df, X_train, X_test, y_train, y_test, df_train, df_test


df, X_train, X_test, y_train, y_test, df_train, df_test = read_data()
today = dt.datetime.combine(dt.date.today(), dt.datetime.min.time())
today = datetime.date(today)
st.header(today)
st.title("Stock Market Prediction using Sentiment Analysis")
st.markdown("Use today top 25 news to predict whether tomorrow's Dow Jones Opening Value will increase or decrease. Our timeline ranges from 8th of August 2008 to the 29th of June 2016.")
st.markdown("More technically, we condensed the top 25 news and implemented a sentiment analysis on a daily basis. As a result, the news of each day are classified along 5 characteristics, namely (1) Objectivity, (2) Subjectivity, (3) Positive, (4) Negative, (5) Neutral.")
st.markdown("On this page, you will find an extract of our dataset alongside a graphical representation of the Dow Jones opening values throughout our timeline.")
st.markdown("On the next page, you will find the necessary tool to discover the trend of the Dow Jones tomorrow morning.")
st.markdown("**Dataset: First Five Rows**")
st.dataframe(df.head())
st.markdown("**Graph: Dow Jones Index Opening Values from 2008 to 2016**")
dow_open = df[['Date','Open']]
dow_open.set_index('Date')
dow_open = dow_open.rename(columns={'Date':'index'}).set_index('index')
st.line_chart(dow_open)
