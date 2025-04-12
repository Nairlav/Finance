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

#Take Order
def take_order(symbol, volume, type, price, stop_loss, take_profit, magic):

    order_type = {
        'Long': mt5.ORDER_TYPE_BUY_LIMIT,
        'Short': mt5.ORDER_TYPE_SELL_LIMIT
    }

    request = {
    "action": mt5.TRADE_ACTION_PENDING,
    "symbol": symbol,
    "volume": volume, # FLOAT
    "type": order_type[type],
    "price": price,
    "sl": stop_loss, # FLOAT
    "tp": take_profit, # FLOAT
    "deviation": 10, # INTERGER
    "magic": magic, # INTERGER
    "comment": "python script open",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
    }
    mt5.order_send(request)
    

#Cancel Order
def cancel_order(id):
    request={
        "action": mt5.TRADE_ACTION_REMOVE,
        "order":id
    }
    mt5.order_send(request)
    
#-------------------------------------------- Connection / Variables

connection()

ticker="EURUSD"
timeframe=mt5.TIMEFRAME_M1  #M1, M5, M15, M30, H1, H4, D1, W1
start=datetime(2024, 1, 19)
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
'''
plt.plot(pivots["Last_High_Value"].loc[((pivots["HH"] == True) | (pivots["LH"] == True))], color='red', marker='x')
plt.plot(pivots["Last_Low_Value"].loc[((pivots["LL"] == True) | (pivots["HL"] == True))], color='green', marker='x')
plt.plot(DATA['close'])
plt.show()
'''
#-------------------------------------------- Strategy
i=0
high_prev=0
low_prev=0
high_actual=[0,0]
low_actual=[0,0]
order_passed=False

while 1:
    s_l=0.786
    entry=0.5
    t_p=0.2
    #Download Bar Data
    end=mt5.symbol_info_tick("EURUSD")
    DATA = get_bar_data(ticker,timeframe,start,end.time)
    pivots=pivotPoints(pivot_variable,DATA)
    order_book=pd.DataFrame(mt5.orders_get())
    print(datetime.utcfromtimestamp(end.time).strftime('%y-%m-%d %H:%M:%S')," | high: ",high_actual,"| low: ",low_actual)

    #--------
    
    # Delete Pending Order if price over TP
    
    if not order_book.empty:
        for id,value in order_book.iterrows(): #    18=tp=fib(t_p)  | value[17]=sl=fib(s_l)
            eend=(value[18]-t_p*value[17]/s_l)/(1-t_p-t_p*(1-s_l)/s_l)
            startt=(value[18]-eend*(1-t_p))/t_p
            if (end.ask>eend or end.ask<startt) and value[6]==2:
                cancel_order(value[0])
            if (end.bid < eend or end.ask>startt) and value[6]==3:
                cancel_order(value[0])
    
    #--------
                
    #Actualization of signals
    if (pivots.iloc[-2]["HL"]==True & pivots.iloc[-1]["HH"]==True) | (pivots.iloc[-2]["LL"]==True & pivots.iloc[-1]["HH"]==True) | (pivots.iloc[-2]["HL"]==True & pivots.iloc[-1]["LH"]==True) | (pivots.iloc[-2]["LL"]==True & pivots.iloc[-1]["LH"]==True) :
        high_actual=[pivots.iloc[-2]["Last_Low_Value"],pivots.iloc[-1]["Last_High_Value"]]
    if (pivots.iloc[-2]["LH"]==True & pivots.iloc[-1]["LL"]==True) | ((pivots.iloc[-2]["HH"]==True & pivots.iloc[-1]["LL"]==True) | pivots.iloc[-2]["LH"]==True & pivots.iloc[-1]["HL"]==True) | ((pivots.iloc[-2]["HH"]==True & pivots.iloc[-1]["HL"]==True)):
        low_actual=[pivots.iloc[-2]["Last_High_Value"],pivots.iloc[-1]["Last_Low_Value"]]
    
    #--------
        
    #Pending Long
    if (high_prev==0 and high_actual!=[0,0]) or (high_prev!=high_actual and high_prev!=0):   
        high_prev=high_actual
        HH= high_actual[1]
        HL= high_actual[0]
        fibo=fibonacci(HL,HH)  
        volume=2.0
        type="Long"
        sl=fibo["Value"].loc[s_l]
        price=fibo["Value"].loc[entry]
        tp=fibo["Value"].loc[t_p]
        magic=234000
        take_order("EURUSD",volume,type,price,sl,tp,magic)
        order_passed=True
        
    #--------
        
    #Pending Short
    if (low_prev==0 and low_actual!=[0,0]) or (low_prev!=low_actual and low_prev!=0):   #Short
        low_prev=low_actual
        LL= low_actual[1]
        LH= low_actual[0]
        fibo=fibonacci(LH,LL)  
        volume=2.0
        type="Short"
        sl=fibo["Value"].loc[s_l]
        price=fibo["Value"].loc[entry]
        tp=fibo["Value"].loc[t_p]
        magic=234000
        take_order("EURUSD",volume,type,price,sl,tp,magic)
        order_passed=True
    
    #--------
        
    #Print Result if pending order past
    if order_passed:
        print("Total Orders: ",mt5.orders_total()) 
        print("Total Positions: ",mt5.positions_total()) 
        print(mt5.orders_get())
        order_book=pd.DataFrame(mt5.orders_get())
        print(order_book)
        order_passed=False
        print("high_actual: ",high_actual,"       low_actual: ",low_actual)

    time.sleep(2)










































mt5.shutdown()