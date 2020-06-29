##########################################################################
#     Libraries
##########################################################################
from datetime import datetime
import joblib
import pandas as pd
import pytz
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from STOCK_PREDICT.data import Data



##########################################################################
#     Application
##########################################################################



data = Data()
df, X_train, X_test, y_train, y_test = data.clean_df()

st.markdown("# Stock Market Prediction using Sentiment Analysis")
st.markdown("**Let's go**")

if __name__ == "__main__":
    #df = read_data()
    main()
