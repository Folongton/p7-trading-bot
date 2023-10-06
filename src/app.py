import streamlit as st
from datetime import datetime
import logging

from src.data.get_data import YahooFinanceAPI as yfapi
from src.models_service.models_service import TensorflowModelService as TFModelService
from src.common.logs import setup_logging

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

config = {
    'ticker': 'MSFT',
    'model': {
        'window': 20,
        'batch_size' : 32,
        
    },
}


# Create a Streamlit app
st.markdown("<h1 style='color: teal;'>Microsoft Stock Prediction App</h1>", unsafe_allow_html=True)


# Get the stock data
today = datetime.today().strftime('%Y-%m-%d')
stock_data_YTD = yfapi.get_daily_data(config['ticker'], start_date='2023-01-01', end_date=today)

# data prep
stock_data_df = stock_data_YTD.tail(config['model']['window'])[::-1]
stock_data_df = stock_data_df[['Adj Close', 'Volume']]

# Calculate the next day's prediction
model_name = 'LSTM_42113_2023_10_03__04_26'
model = TFModelService.load_model(model_name=model_name, logger=logger)
scalers = TFModelService.load_scalers(model_name=model_name, logger=logger)

next_day_prediction = TFModelService.model_forecast(model=model, 
                                                    df=stock_data_df,
                                                    window_size=config['model']['window'],
                                                    scalers=scalers,
                                                    verbose=False)

# Display the prediction
st.write(f'Tomorrow Microsoft will close at: $ {round(float(next_day_prediction),2)}')

# display the model name
st.write(f'Model name: {model_name}')

# name chart in  teal "Microsoft Stock Price YTD"
st.markdown("<h2 style='color: teal;'>Microsoft Stock Price YTD</h2>", unsafe_allow_html=True)
st.line_chart(stock_data_YTD['Adj Close'])

# display table of the last 20 days data and center it in the page
st.table(stock_data_df[['Adj Close', 'Volume']].rename(columns={'Adj Close': 'Price'}))



