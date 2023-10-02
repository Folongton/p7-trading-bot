import streamlit as st
from datetime import datetime
import logging
import os
import sys

# # ---------- Add project root to path ----------
# from dotenv import find_dotenv, load_dotenv
# load_dotenv(find_dotenv(), verbose=True) # Example:  AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY")
# PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
# sys.path.append(PROJECT_ROOT)
# # ----------------------------------------------

# from src.data.get_data import YahooFinanceAPI as yfapi
# from src.models_service.models_service import TensorflowModelService as TFModelService
# from src.common.logs import setup_logging

# logger = setup_logging(logger_name=__name__,
#                         console_level=logging.INFO, 
#                         log_file_level=logging.INFO)

# config = {
#     'ticker': 'MSFT',
#     'model': {
#         'window': 20,
#         'batch_size' : 32,
        
#     },
# }


# Create a Streamlit app

st.markdown("<h1 style='color: teal;'>Microsoft Stock Prediction App</h1>", unsafe_allow_html=True)


# # Get the stock data
# today = datetime.today().strftime('%Y-%m-%d')
# stock_data_df = yfapi.get_daily_data(config['ticker'], start_date='2023-01-01', end_date=today)

# # data prep
# stock_data_df = stock_data_df.tail(config['model']['window'])[::-1]
# stock_data_df = stock_data_df[['Adj Close', 'Volume']]

# # Calculate the next day's prediction
# model = TFModelService.load_model(model_name='LSTM_42113_2023-09-20--15-58', logger=logger)
# scalers = TFModelService.load_scalers(model_name='LSTM_42113_2023-09-20--15-58', logger=logger)

# next_day_prediction = TFModelService.model_forecast(model=model, 
#                                                     df=stock_data_df,
#                                                     window_size=config['model']['window'],
#                                                     scalers=scalers,
#                                                     verbose=True)

# # Display the prediction
# st.write(f'Tomorrow Microsoft price will close at: $ {round(float(next_day_prediction),2)}')
