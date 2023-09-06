from dotenv import find_dotenv, load_dotenv
import os
import json as j
import pandas as pd
import requests as r
import time
from pathlib import Path
load_dotenv(find_dotenv(), verbose=True)


AV_KEY = os.environ.get("ALPHA_VANTAGE_FREE_KEY")
CALLS_PER_MINUTE = 75
SLEEP_TIME = 60/CALLS_PER_MINUTE + (60/CALLS_PER_MINUTE)*0.1 # 75 requests per minute with 10% buffer

project_dir = Path(__file__).resolve().parents[2]
DATA_DIR = f'{project_dir}\data\\00_raw\daily_full'

class AlphaVantageAPI:
    @staticmethod
    def get_daily_data(ticker, apikey, full_or_compact='full'):
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
    def get_daily_data_for_list(ticker_list, directory, apikey, full_or_compact, sleep_time):
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

class CSVsLoader:
    @staticmethod
    def load_daily(ticker, directory=DATA_DIR):
        '''
        Loads the daily data from a csv file.
        IN: ticker, directory
        OUT: pandas dataframe
        '''
        df = pd.read_csv(os.path.join(directory, f'{ticker}-daily-full.csv'), index_col=0, parse_dates=True)
        df.sort_index(ascending=True, inplace=True)
        df.name = ticker
        return df


if __name__ == '__main__':
    AlphaVantageAPI.get_daily_data_for_list(['VZ', 'INTC', 'ABBV', 'F', 'JNJ'], DATA_DIR,  AV_KEY, 'full', SLEEP_TIME) # Value stocks
    AlphaVantageAPI.get_daily_data_for_list(['BABA', 'AMZN', 'MSFT', 'TSLA', 'GOOGL'], DATA_DIR, AV_KEY, 'full', SLEEP_TIME) # Growth stocks