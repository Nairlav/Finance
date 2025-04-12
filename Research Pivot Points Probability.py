#--------------------------------------------
#Librairies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import ta as ta
import warnings
from collections import deque
from datetime import datetime, timedelta
warnings.simplefilter(action='ignore', category=FutureWarning)
#------------------------
#Variables DATA
ticker="BTC-USD"
timeframe=["1m", "2m", "5m", "15m", "30m", "60m","90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"] #1m only for 1d and <1d for 60d
period=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] #60d is max 
#-------------------------------------------------------------- Function
#Get DATA
def get_data(ticker, timeframe, nbr_day):
    DATA=pd.DataFrame()
    start_date = datetime.today().date()-timedelta(days=nbr_day)
    end_date = start_date + timedelta(days=1)
    
    for i in range(0, nbr_day):
        start_date = end_date
        end_date = start_date + timedelta(days=1)
        DATA1 = yf.download(tickers=ticker, interval=timeframe, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        DATA = pd.concat([DATA, DATA1])
        
    return DATA

#Pivot Points
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
    keyHigh = 'High'
    keyLow = 'Low'
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
    Pivot_Points = Pivot_Points.drop(['H', 'L'], axis=1)


    Pivot_Points['Last_High_Value'].fillna(method='ffill', inplace=True)
    Pivot_Points['Last_Low_Value'].fillna(method='ffill', inplace=True)
    H=Pivot_Points['Last_High_Value']
    L=Pivot_Points['Last_Low_Value']
    Pivot_Points=Pivot_Points[["HH","LH","LL","HL"]].fillna(False)
    Pivot_Points['Last_High_Value']=H
    Pivot_Points['Last_Low_Value']=L

    to_drop = []  # List to hold indices of rows to be dropped

    # Assuming 'HH', 'LH', 'HL', and 'LL' are column names
    for i in range(len(Pivot_Points) - 1):
        condition1 = ((Pivot_Points['HH'].iloc[i] & Pivot_Points['HH'].iloc[i+1]) or
                    (Pivot_Points['LH'].iloc[i] & Pivot_Points['HH'].iloc[i+1]) or
                    (Pivot_Points['HL'].iloc[i] & Pivot_Points['LL'].iloc[i+1]) or
                    (Pivot_Points['LL'].iloc[i] & Pivot_Points['LL'].iloc[i+1]))
        condition2 = ((Pivot_Points['HH'].iloc[i] & Pivot_Points['LH'].iloc[i+1]) or
                    (Pivot_Points['HL'].iloc[i] & Pivot_Points['HL'].iloc[i+1]) or
                    (Pivot_Points['LH'].iloc[i] & Pivot_Points['LH'].iloc[i+1]) or
                    (Pivot_Points['LL'].iloc[i] & Pivot_Points['HL'].iloc[i+1]))
        
        if condition1:
            to_drop.append(i)
        elif condition2:
            to_drop.append(i+1)

        # Get the actual index labels for the positions in 'to_drop'
    labels_to_drop = Pivot_Points.index[to_drop]

    # Drop all the rows in one go
    Pivot_Points.drop(index=labels_to_drop, inplace=True)

    return Pivot_Points

#Time Diff btw pivots
def time_diff(pivots,timeframe_use):
    timeframe_minutes = int(timeframe_use[:-1])
    a=[0]
    for i in range(1,len(pivots)):
        b=int((pivots.index[i]-pivots.index[i-1]).total_seconds()/(60*timeframe_minutes))
        if b==0:
            b=1
        a.append(b)
    pivots["Time_Diff"]=np.array(a)  
    return pivots

#Coefficient with Trend
def pivot_coeff(pivots):
    if pivots['Last_High_Value'].isna().iloc[0]:
        trend="low"
    else:
        trend="high"
    high=[0,0]
    low=[0,0]
    for i in range(2,len(pivots)):
        if trend=="high":
            high.append(pivots["Last_High_Value"].iloc[i]-pivots["Last_High_Value"].iloc[i-2])
            low.append(0)
            trend="low"
            continue
        if trend=="low":
            low.append(pivots["Last_Low_Value"].iloc[i]-pivots["Last_Low_Value"].iloc[i-2])
            high.append(0)
            trend="high" 
            continue  
    pivots["Next_High_Trend"]=np.array(high)
    pivots["Next_Low_Trend"]=np.array(low)
    return pivots

#Gaussian and Histogram
def Gaussian_Histo(coefficient):

    mean=coefficient.mean()
    std=coefficient.std()

    #Min=coefficient.min()
    #Max=coefficient.max()

    Min=mean-3*std
    Max=mean+3*std
    coefficient= coefficient[(coefficient <= Max) & (coefficient >= Min) & (coefficient!=0)]
    print("---------------")
    print("Mean:",mean,"\nStd:",std)
    print("Nbr Coeff:",len(coefficient))
    print("---------------")

    plt.hist(coefficient,bins=int(len(coefficient)/10),density=True)
    X=np.linspace(Min,Max,100)
    Gausian=norm.pdf(X,mean,std)

    plt.plot(X,Gausian)
    plt.title("Histogram Pivot Points")
    plt.show()

    return mean,std

#Get State
def get_state(pivots,coefficient,std,interval_std):
    ranges = np.array(interval_std) * std
    a = np.zeros(coefficient.shape)

    for idx, (low, high) in enumerate(zip(ranges[:-1], ranges[1:]), start=1):
        
        a[(coefficient > low) & (coefficient <= high)] = idx
    a[coefficient == 0] = 0

    pivots["State"]=np.array(a)
    pivots["State"]=pivots["State"].astype(int)
    return pivots

#Get Matrix with Probability
def Get_Matrix(pivots,interval_std,size_X,size_Y):
    matrix=np.zeros((size_X,size_Y))

    Index=[]
    for i in range(1,len(interval_std)):
        for j in range(1,len(interval_std)):
            Index.append(f"{i}_{j}")
    matrix=pd.DataFrame(matrix,columns=Index,index=Index)
    matrix=matrix.astype(int)

    for i in range(5, len(pivots["State"])):
        
        state_i_3 = pivots["State"].iloc[i-3]
        state_i_2 = pivots["State"].iloc[i-2]
        state_i_1 = pivots["State"].iloc[i-1]
        state_i_0 = pivots["State"].iloc[i-0]

        x = f"{state_i_3}_{state_i_2}"
        y = f"{state_i_1}_{state_i_0}" 
        
        if state_i_0==0 or state_i_2==0 or state_i_1==0 or state_i_3==0 :
            continue

        matrix[y].loc[x] += 1
    
    total=matrix.sum().sum()
    print("---------------")
    print("Total Value:",total)
    print("---------------")

    matrix["Total"]=matrix.sum(axis=1)
    matrix["Probability"]=matrix["Total"]/total
    matrix.loc["Total"]=matrix[matrix.columns].sum()

    return total,matrix

#-------------------------------------------------------------- Analyze
#-------------- Parameters
pivot_parameter=25
Timeframe=timeframe[0]
nbr_day=29
RSI_parameter=15
Reload_Data=False

print("#--------------")
print("VARIABLES:")
print("Ticker:",ticker)
print("Period:",nbr_day,"day(s)")
print("Timeframe:",Timeframe)
print("Pivot Point Value:",pivot_parameter)
print("RSI Value:",RSI_parameter)
print("#--------------")

#-------------- Get Data
if Reload_Data:
    DATA=get_data(ticker,Timeframe,nbr_day)
    DATA.to_csv('DATA.csv',index=True)
DATA=pd.read_csv("Data.csv", index_col=0, parse_dates=True)
print(DATA.head())
DATA=DATA[:2000]
#-------------- Get Pivot Points
pivots=pivotPoints(pivot_parameter,DATA)

#-------------- Show DATA in graph

plt.plot(pivots["Last_High_Value"].loc[((pivots["HH"] == True) | (pivots["LH"] == True))], color='red', marker='x')
plt.plot(pivots["Last_Low_Value"].loc[((pivots["LL"] == True) | (pivots["HL"] == True))], color='green', marker='x')
plt.plot(DATA["Close"])
plt.title(ticker+" with Pivot Points")
plt.show()
#-------------- Get Coeffficient of Variation

#timeframe_use=timeframe[0]
#pivots=time_diff(pivots,timeframe_use)
pivots=pivot_coeff(pivots)

#-------------- Show Histogram with Gaussian

coefficient=np.array(pivots["Next_High_Trend"])+np.array(pivots["Next_Low_Trend"])
mean,std=Gaussian_Histo(coefficient)

#-------------- Get State for each coefficient
interval_std=[-2,-2/3, -1/6, 1/6, 2/3, 2]
pivots=get_state(pivots,coefficient,std,interval_std)
print(pivots[["Next_High_Trend","Next_Low_Trend","State"]].head())

#-------------- Get State for each coefficient
size_X=(len(interval_std)-1)**2
size_Y=(len(interval_std)-1)**2
total,matrix=Get_Matrix(pivots,interval_std,size_X,size_Y)
print(matrix)
print("---------------")
matrix2=pd.DataFrame(matrix.loc["Total"]/total)
print(matrix2)
print("---------------")

#-------------- Get Markov matrix
value=["Bull","Range","Bear"]
markov_matrix=pd.DataFrame(np.zeros((3,3)),index=value,columns=value)
Bull=["3_3","2_3","3_2"]
Bear=["1_1","2_1","1_2"]
Range=["3_1","1_3","2_2",]

for i in matrix.index[:-1]:
    for j in matrix.index[:-1]:

        #Bull
        if i in Bull and j in Bull:
            markov_matrix["Bull"].loc["Bull"]+=matrix[j].loc[i]
        if i in Bull and j in Range:
            markov_matrix["Bull"].loc["Range"]+=matrix[j].loc[i]
        if i in Range and j in Bull:
            markov_matrix["Range"].loc["Bull"]+=matrix[j].loc[i]
        
        #Bear
        if i in Bear and j in Bear:
            markov_matrix["Bear"].loc["Bear"]+=matrix[j].loc[i] 
        if i in Range and j in Bear:
            markov_matrix["Range"].loc["Bear"]+=matrix[j].loc[i] 
        if i in Bear and j in Range:
            markov_matrix["Bear"].loc["Range"]+=matrix[j].loc[i]  
        
        #Range
        if i in Bear and j in Bull:
            markov_matrix["Bear"].loc["Bull"]+=matrix[j].loc[i]
        if i in Range and j in Range:
            markov_matrix["Range"].loc["Range"]+=matrix[j].loc[i]  
        if i in Bull and j in Bear:
            markov_matrix["Bull"].loc["Bear"]+=matrix[j].loc[i]

A = markov_matrix.sum(axis=0) / markov_matrix.sum().sum()

for i in markov_matrix.index:
    markov_matrix.loc[i]=markov_matrix.loc[i]/markov_matrix.loc[i].sum()
print("Markov Matrix:")
print(markov_matrix)



#-------------------------------------------------------------- 