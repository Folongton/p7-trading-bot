import pandas as pd
import os
import sys 
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter


project_dir = Path(__file__).resolve().parents[2]
DATA_DIR = f'{project_dir}\data\\00_raw\daily_full'

plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('Solarize_Light2')

class AlphaVantagePlots:
    @staticmethod
    def calc_prct_gain(series):
        return series.apply(lambda x: (x - series.iloc[0])/series.iloc[0])
    
    @staticmethod
    def plot_prct_gain(stocks, from_date, to_date=datetime.now()-timedelta(days=1) , title='Percent Gain'):
        plt.figure(figsize=(15, 5))
        for stock in stocks:
            df = pd.read_csv(os.path.join(DATA_DIR, rf'{stock}-daily-full.csv'), index_col=0, parse_dates=True)
            df.index.rename('Date', inplace=True)
            df = df.copy()[::-1]
            df = df.loc[from_date:to_date].copy(deep=True)

            AlphaVantagePlots.calc_prct_gain(df['5. adjusted close']).plot()

        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        plt.gca().yaxis.tick_right()
        plt.legend(stocks)
        plt.title(title)
        plt.legend(stocks)
        plt.show()  