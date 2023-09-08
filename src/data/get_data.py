import os
import json as j
import pandas as pd
import requests as r
import time
import logging
from src.common.logs import setup_logging

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), verbose=True)

from src.common.globals import G
from pathlib import Path

AV_KEY = os.environ.get("ALPHA_VANTAGE_FREE_KEY")
CALLS_PER_MINUTE = 75
SLEEP_TIME = 60/CALLS_PER_MINUTE + (60/CALLS_PER_MINUTE)*0.1 # 75 requests per minute with 10% buffer
# MUST be updated for cheapest API, since AV no longer offers free API with 60 calls per minute. see here : https://www.alphavantage.co/premium/

DATA_DIR_DAILY_FULL = G.raw_daily_full_dir
NASDAQ_ALL = G.all_nasdaq_tickers

logger = setup_logging(logger_name=__name__,
                        console_level=logging.INFO, 
                        log_file_level=logging.INFO)


class AlphaVantageAPI:
    @staticmethod
    def get_daily_data(ticker, apikey=AV_KEY, full_or_compact='full'):
        '''
        https://www.alphavantage.co/documentation/#dailyadj
        Returns formatted data from AlphaVantage API
        IN: API Parameters
        OUT: Json Data'''
        url_json = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={apikey}&outputsize={full_or_compact}'
        response = r.get(url_json)
        j_response = response.json()
        formated_response = j.dumps(j_response, indent=4)
        return formated_response
    
    @staticmethod
    def get_daily_data_for_list(ticker_list, directory, full_or_compact,  apikey=AV_KEY, sleep_time=SLEEP_TIME):
        '''
        https://www.alphavantage.co/documentation/#dailyadj
        Saves Data from AlphaVantage API to a folder in 1 csv file per ticker
        IN: API Parameters, directory to save data to
        OUT: None - saves data to directory'''
        for ticker in ticker_list:
            print(f'{ticker_list.index(ticker)/len(ticker_list)*100:.2f}%')

            if os.path.exists(os.path.join(directory, f'{ticker}-daily-{full_or_compact}.csv')):
                pass
            elif os.path.exists(os.path.join(directory, f'{ticker}.json')):
                pass
            else:
                data = AlphaVantageAPI.get_daily_data(ticker,apikey,full_or_compact)
                data = j.loads(data)
                try:
                    df = pd.DataFrame(data['Time Series (Daily)']).T
                    df.index.name = 'Date'
                    df.to_csv(os.path.join(directory, f'{ticker}-daily-{full_or_compact}.csv'))
                    time.sleep(sleep_time)
                except KeyError:
                    print(f'No data for {ticker}. Saving Json file')
                    print(data)
                    with open(os.path.join(directory, f'{ticker}.json'), 'w') as f:
                        j.dump(data, f)
                    continue
        print('100.00%')

    @staticmethod
    def get_intraday_data(ticker, apikey, interval='5min', adjusted='false', outputsize='compact', datatype='json'):
        '''
        https://www.alphavantage.co/documentation/#intraday
        Returns json data from AlphaVantage API
        IN: API Parameters
        OUT: Json Data
        '''
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&apikey={apikey}&adjusted={adjusted}&outputsize={outputsize}&datatype={datatype}'
        response = r.get(url)

        if response.status_code == 200:
            data = j.loads(response.text)
            return data
        else:
            print(f'Error: {response.status_code}')
            return None
    
    @staticmethod
    def get_intraday_data_for_list(tickers_list, directory, apikey, sleep_time, interval='5min', adjusted='false', outputsize='compact', datatype='json'):
        '''
        https://www.alphavantage.co/documentation/#intraday
        This function takes in the following parameters:
        symbol: The name of the equity you want to retrieve intraday data for.
        interval: The time interval between two consecutive data points in the time series (e.g. "1min", "5min", "15min", "30min","60min").
        apikey: Your API key for the Alpha Vantage API.
        adjusted: An optional boolean/string value ('true' or 'false') indicating whether the output time series should be adjusted by historical split 
                  and dividend events (default is 'true').
        outputsize: An optional string value indicating the size of the output time series (default is "compact", which returns the latest 100 data points).
        datatype: An optional string value indicating the data format of the output (default is "json").
        '''                                                                                                                                                                      
        for ticker in tickers_list:
            print(f'{round(tickers_list.index(ticker)/(len(tickers_list)-1)*100,2)}% completed')

            if os.path.exists(os.path.join(directory, f'{ticker}-1day-5min.csv')):
                pass
            elif os.path.exists(os.path.join(directory, f'{ticker}.json')):
                pass
            else:
                data = AlphaVantageAPI.get_intraday_data(ticker, interval=interval, apikey=apikey, adjusted=adjusted, outputsize=outputsize, datatype=datatype)
                if data:
                    try:
                        df = pd.DataFrame(data['Time Series (5min)']).T
                        df.index = pd.to_datetime(df.index)
                        df.index.name = 'Date-Time'
                        df.to_csv(os.path.join(directory, f'{ticker}-1day-5min.csv'))
                        time.sleep(sleep_time) #to avoid API limit of 75 calls per minute
                        
                    except KeyError:
                        print(f'No data for {ticker}. Saving Json file')
                        print(data)
                        with open(os.path.join(directory, f'{ticker}.json'), 'w') as f:
                            j.dump(data, f)
                        continue
            print (r"100.00% completed")
                    
    @staticmethod
    def get_intraday_extended_data(ticker, key, sleep_time, directory, interval='60min',slice='year1month1'):
        '''
        Read here : https://www.alphavantage.co/documentation/#intraday-extended
        Gets the intraday data for a given ticker and saves it as a csv file.
        IN: API parameters
        OUT: None - saves the data as a csv file.
        '''
        if slice == '2yearData':
            months = ['1','2','3','4','5','6','7','8','9','10','11','12']
            years = ['1','2']
            with open(os.path.join(directory, f'{ticker}-2years-hourly.csv'), 'w') as f:
                for year in years:
                    for month in months:
                        slice = f'year{year}month{month}'
                        url_json = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval={interval}&slice={slice}&apikey={key}'
                        response = r.get(url_json)
                        f.write(response.text)
                        time.sleep(sleep_time) 
        else:
            url_json = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval={interval}&slice={slice}&apikey={key}'
            response = r.get(url_json)
            with open(os.path.join(directory, f'{ticker}-30days-hourly.csv'), 'w') as f:
                f.write(response.text)

class CSVsLoader(pd.DataFrame):
    def __init__(self, ticker , directory=DATA_DIR_DAILY_FULL , *args, **kwargs):
        # Create a DataFrame object (empty)
        super().__init__(*args, **kwargs)
        self.ticker = ticker
        self.directory = directory
        self.load_daily(ticker=self.ticker, directory=self.directory)

    def load_daily(self, ticker, directory):
        '''
        Loads the daily data from a csv file.
        IN: ticker, directory
        OUT: pandas dataframe
        '''
        full_path = os.path.join(directory, f'{ticker}-daily-full.csv')
        parts = Path(full_path).parts[-5:]
        for_logs = os.path.join(r'..', parts[0], parts[1], parts[2], parts[3], parts[4])

        df = pd.read_csv(full_path, index_col=0, parse_dates=True, date_format='yyyy-mm-dd' )
        df.index = pd.to_datetime(df.index)
        df.sort_index(ascending=True, inplace=True)
        df.name = ticker

        self.__dict__.update(df.__dict__)

        logger.info(f'Loaded "{for_logs}". Number data points {df.shape[0]}. From "{df.index[0]}" to "{df.index[-1]}"')


    def prep_AV_data(self):
        self.drop(columns=['1. open', '2. high', '3. low', '4. close', '7. dividend amount', '8. split coefficient'], inplace=True)
        self.rename(columns={'5. adjusted close': 'Adj Close', '6. volume': 'Volume'}, inplace=True)
        self.index.name = 'Date'
        self.sort_index(ascending=True, inplace=True)
        self.dropna(inplace=True, subset=['Adj Close'])
        return self

    def save_data(self, directory=''):
        '''
        Saves the data as a csv file into 03_processed folder.'''
        self.to_csv(os.path.join(directory, f'{self.name}-daily-full.csv'))

if __name__ == '__main__':
    AlphaVantageAPI.get_daily_data_for_list(['VZ', 'INTC', 'ABBV', 'F', 'JNJ'], DATA_DIR_DAILY_FULL,  AV_KEY, 'full', SLEEP_TIME) # Value stocks
    AlphaVantageAPI.get_daily_data_for_list(['BABA', 'AMZN', 'MSFT', 'TSLA', 'GOOGL'], DATA_DIR_DAILY_FULL, AV_KEY, 'full', SLEEP_TIME) # Growth stocks
    AlphaVantageAPI.get_daily_data_for_list(NASDAQ_ALL, DATA_DIR_DAILY_FULL, AV_KEY, 'full', SLEEP_TIME) # All NASDAQ stocks
