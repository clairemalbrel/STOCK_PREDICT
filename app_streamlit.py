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
import csv
from pathlib import Path
import platform
import altair as alt

##########################################################################
#     Application
##########################################################################

@st.cache
def read_data():
  data = Data()
  df, X_train, X_test, y_train, y_test, df_train, df_test = data.clean_df()
  return df, X_train, X_test, y_train, y_test, df_train, df_test

df, X_train, X_test, y_train, y_test, df_train, df_test = read_data()
directory = str(Path.home()) + '/code/stock_market/STOCK_PREDICT/'
if platform.system() == 'Windows':
  directory = directory.replace('\\' ,'/')
df_prediction = pd.read_csv(f'{directory}data2/final.csv')

#This is today's date
today = dt.datetime.combine(dt.date.today(), dt.datetime.min.time())
today = datetime.date(today)
st.header(today)

#This is the header
st.markdown("<h1 style='text-align: center; color: DarkBlue;'>Stock Market Prediction using Sentiment Analysis</h1>", unsafe_allow_html=True)

st.text("")
st.text("")

analysis = st.sidebar.selectbox("Menu",['Project Presentation', 'Prediction Tool'])

if analysis == "Project Presentation":
  #This is the abstract
  st.markdown("<h1 style='text-align: left; color: CornflowerBlue;'>Abstract</h1>",unsafe_allow_html=True)
  st.markdown("Use today top 25 news to predict whether tomorrow's Dow Jones Opening Value will increase or decrease. Our timeline ranges from 8th of August 2008 to the 29th of June 2016.")
  st.markdown("More technically, we condensed the top 25 news and implemented a sentiment analysis on a daily basis. As a result, the news of each day are classified along 5 characteristics, namely (1) Objectivity, (2) Subjectivity, (3) Positive, (4) Negative, (5) Neutral.")
  st.markdown("On this page, you will find an extract of our dataset alongside a graphical representation of the Dow Jones opening values throughout our timeline.")
  st.markdown("On the next page, you will find the necessary tool to discover the trend of the Dow Jones tomorrow morning and the sentiment scores of the day before.")

  #This is the line plof of the Dow Jones
  st.markdown("<h1 style='text-align: left; color: CornflowerBlue;'>Graph: Dow Jones Index Opening Values from 2008 to 2016</h1>",unsafe_allow_html=True)
  dow_open = df[['Date','Open']]
  dow_open.set_index('Date')
  dow_open = dow_open.rename(columns={'Date':'index'}).set_index('index')
  st.line_chart(dow_open)

  #This is an extract of our datasets
  st.markdown("<h1 style='text-align: left; color: CornflowerBlue;'>Dataset: First Five Rows</h1>",unsafe_allow_html=True)
  st.dataframe(df.head())

if analysis == 'Prediction Tool':
  #This is the slider to select the day you want
  option = st.selectbox('Which day do you like to consult?',df_prediction['Date'])

  st.text("")

  #This gives you the prediction of our model: Rise/Decrease
  name = 'The Dow Jones will ' + df_prediction['prediction'][0] + ' at the opening.'
  if st.button('CLICK HERE TO SEE OUR PREDICTION'):
    st.markdown(name)

  st.text("")

  #This gives the bar chart of the Sentiment Scores the day before the prediction
  st.markdown("<h1 style='text-align: left; color: CornflowerBlue;'>Yesterday's Sentiment Scores</h1>",unsafe_allow_html=True)
  sentiment_score = df_prediction[['Subjectivity','Objectivity','Positive','Negative','Neutral']]
  st.bar_chart(sentiment_score.transpose())
