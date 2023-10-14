import streamlit as st
from datetime import datetime, timedelta
import logging

from src.data.get_data import YahooFinanceAPI as yfapi
from src.models_service.models_service import TensorflowModelService as TFModelService
from src.features.build_features import FeatureEngineering as FE
from src.common.plots import Visualize as V
from src.common.logs import setup_logging

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)

config = {
    'ticker': 'MSFT',
    'model': {
        'window': 120,
        'batch_size' : 128,
        
    },
}


# Create a Streamlit app
st.markdown("<h1 style='color: teal;'>Microsoft Stock Prediction App</h1>", unsafe_allow_html=True)


# Get the stock data
today = datetime.today().strftime('%Y-%m-%d')
stock_data_YTD = yfapi.get_daily_data(config['ticker'], start_date='2023-01-01', end_date=today)
stock_data_df = stock_data_YTD[stock_data_YTD.index > '2023-01-01'][::-1]

# Prepare the data for the model
df_test_X = stock_data_df[['Adj Close', 'Volume']]
df_test_X = FE.rename_shifted_columns(df_test_X)
df_test_y = stock_data_df['Adj Close']


# Calculate predictions and plot Buy and Sell signals
model_name = 'MSFT_LSTM_W20_SBS5500_B32_E5_P42113_2023_10_10__21_04'
model = TFModelService.load_model(model_name=model_name, logger=logger)
scalers = TFModelService.load_scalers(model_name=model_name, logger=logger)
window_size = TFModelService.get_window_size_from_model_name(model._name)

results = TFModelService.model_forecast(model=model, 
                                        df=df_test_X,
                                        window_size=window_size,
                                        scalers=scalers,
                                        verbose=False)
# Create a plot of the predictions
df_test_plot_y = TFModelService.prep_test_df_shape(df_test_y, window_size)

fig_path = V.plot_series(x=df_test_plot_y.index,  # as dates
                        y=(df_test_plot_y, results),
                        model_name=model._name,
                        title=f'Predictions {model._name}',
                        xlabel='Date',
                        ylabel='Price',
                        legend=['Actual', 'Predicted'],
                        show=False)

# display the model name
st.write(f'Model name: {model_name}')

# name chart in  teal "Microsoft Stock Price YTD"
st.markdown(f"<h2 style='color: teal;'>Buy and Sell signals for {config['ticker']}</h2>", unsafe_allow_html=True)

# load the figure and center it in the page 
st.image(fig_path)


# display table of the last 20 days data and center it in the page
st.table(stock_data_df[['Adj Close', 'Volume']].head(20).rename(columns={'Adj Close': 'Price'}))



