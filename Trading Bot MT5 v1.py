#--------------------------------------------
#Librairies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta #https://github.com/twopirllc/pandas-ta
from collections import deque
import MetaTrader5 as mt5
import plotly.express as px
import time
from backtesting import Backtest, Strategy
#--------------------------------------------


def connection():

    login=1
    password=2
    server=3

    mt5.initialize()
    mt5.login(login,password,server)

    account_info = mt5.account_info()
    print('Name: ', account_info.name)
    print('Server: ', account_info.server)
    print('Balance: ',account_info.balance)
    print('Profit: ',account_info.profit)
    print('Equity: ',account_info.equity)

#Bars Data Open, High, Low, Close,
def get_bar_data(ticker,timeframe,start,end):
    DATA=pd.DataFrame(mt5.copy_rates_range(ticker,timeframe,start,end))
    DATA["time"]=DATA["time"].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%y-%m-%d %H:%M'))
    #DATA.set_index("time",inplace=True)
    return DATA[:-1]

#Ticks Data Bid, Ask
def get_tick_data(ticker,start,end,flags):
    DATA=pd.DataFrame(mt5.copy_ticks_range(ticker,start,end,flags))
    DATA["time"]=DATA["time"].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%y-%m-%d %H:%M'))
    DATA["Delta"]=DATA["ask"]-DATA["bid"]
    #DATA.set_index("time",inplace=True)
    return DATA[:-1]

#Fibonacci 1 / 0.786 / 0.618 / 0.5 / 0.382 / 0.236 / 0
def fibonacci(start,end):
    tab=[]
    Fibonacci_Tab=[1,0.786,0.618,0.5,0.382,0.236,0.2,0]
    variation=end-start
    for value in Fibonacci_Tab:
        tab.append([value,start+variation*(1-value)])
    return pd.DataFrame(tab,columns=["Ratio","Value"]).set_index("Ratio")

#Pivot Points HH,LH,LL,HL
def clean_deque(i, k, deq, df, key, isHigh):
    if deq and deq[0] == i - k:
        deq.popleft()
    if isHigh:
        while deq and df.iloc[i][key] > df.iloc[deq[-1]][key]:
            deq.pop()
    else:
        while deq and df.iloc[i][key] < df.iloc[deq[-1]][key]:
            deq.pop()
def pivotPoints(pivot=None,data=None):
    data['H'] = False
    data['L'] = False
    keyHigh = 'high'
    keyLow = 'low'
    win_size = pivot * 2 + 1
    deqHigh = deque()
    deqLow = deque()
    max_idx = 0
    min_idx = 0
    i = 0
    j = pivot
    pivot_low = None
    pivot_high = None
    for index, row in data.iterrows():
        if i < win_size:
            clean_deque(i, win_size, deqHigh, data, keyHigh, True)
            clean_deque(i, win_size, deqLow, data, keyLow, False)
            deqHigh.append(i)
            deqLow.append(i)
            if data.iloc[i][keyHigh] > data.iloc[max_idx][keyHigh]:
                max_idx = i
            if data.iloc[i][keyLow] < data.iloc[min_idx][keyLow]:
                min_idx = i
            if i == win_size-1:
                if data.iloc[max_idx][keyHigh] == data.iloc[j][keyHigh]:
                    data.at[data.index[j], 'H'] = True
                    pivot_high = data.iloc[j][keyHigh]
                if data.iloc[min_idx][keyLow] == data.iloc[j][keyLow]:
                    data.at[data.index[j], 'L'] = True
                    pivot_low = data.iloc[j][keyLow]
        if i >= win_size:
            j += 1
            clean_deque(i, win_size, deqHigh, data, keyHigh, True)
            clean_deque(i, win_size, deqLow, data, keyLow, False)
            deqHigh.append(i)
            deqLow.append(i)
            pivot_val = data.iloc[deqHigh[0]][keyHigh]
            if pivot_val == data.iloc[j][keyHigh]:
                data.at[data.index[j], 'H'] = True
                pivot_high = data.iloc[j][keyHigh]
            if data.iloc[deqLow[0]][keyLow] == data.iloc[j][keyLow]:
                data.at[data.index[j], 'L'] = True
                pivot_low = data.iloc[j][keyLow]

        data.at[data.index[j], 'Last_High_Value'] = pivot_high
        data.at[data.index[j], 'Last_Low_Value'] = pivot_low
        i = i + 1
    
    pivots=data[["L","H",'Last_Low_Value','Last_High_Value']].loc[~((data['L'] == False) & (data['H'] == False))]
    H=pivots[["H", 'Last_High_Value']].loc[pivots["H"]]
    L=pivots[["L", 'Last_Low_Value']].loc[pivots["L"]]
        
    H['HH'] = H['Last_High_Value'].diff().gt(0)
    H['LH'] = ~H['HH']
    H.at[H.index[0], 'HH'] = True
    H.at[H.index[0], 'LH'] = False
    L['LL'] = L['Last_Low_Value'].diff().lt(0)
    L['HL'] = ~L['LL']
    L.at[L.index[0], 'LL'] = True
    L.at[L.index[0], 'HL'] = False
    Pivot_Points = pd.concat([H, L])
    Pivot_Points.sort_index(inplace=True)

    Pivot_Points['Last_High_Value'].ffill(inplace=True)
    Pivot_Points['Last_Low_Value'].ffill(inplace=True)

    Pivot_Points = Pivot_Points.drop(['H', 'L'], axis=1).fillna(False)

    return Pivot_Points

#-------------------------------------------- Connection / Variables

connection()

ticker="EURUSD"
timeframe=mt5.TIMEFRAME_M1  #M1, M5, M15, M30, H1, H4, D1, W1
start=datetime(2023, 12, 19)
end=mt5.symbol_info_tick("EURUSD").time
flags=mt5.COPY_TICKS_ALL    #with get ticks data
pivot_variable=10

#-------------------------------------------- Get Data

#download Bar Data
DATA = get_bar_data(ticker,timeframe,start,end)
pivots=pivotPoints(pivot_variable,DATA)

print(DATA)
print(pivots)

#-------------------------------------------- Plot

class strategy(Strategy):
    def init(self):
        return 0

    def next(self):
        if 1:
            self.buy(sl=.92 * self.data.Close[-1],)

backtest = Backtest(DATA, strategy, commission=.002)
backtest.run()

#backtest.optimize(sma1=[5, 10, 15], sma2=[10, 20, 40],constraint=lambda p: p.sma1 < p.sma2)

