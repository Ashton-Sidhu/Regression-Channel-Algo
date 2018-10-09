# -*- coding: utf-8 -*-
"""
"""

import matplotlib.pyplot as plt
from numpy import log, polyfit, sqrt, std, subtract
from datetime import datetime
import pandas as pd
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import os

yf.pdr_override() # <== that's all it takes :-)


start_date = '2008-10-01' 
end_date = datetime.now() 

symbol = 'V' #ticker of the stock you want to trade (saved as "ticker.csv")
entryZscore = 1 #gauge of the channel
exitZscore = -1 #exit level (can be from 0 to -number). Notice that the entry and the exit have their signs "inverted"
window = 27 #lookback
shorts = 0 #shorts=0 means no shorting, long only
regression = 1 # the mid-line is a linear regression 
complex_entrance = 0
forbidden = 1 # automatic exit
delay = 1 #1 for instant execution, 2 for one day delay
tcost=10/10000*0 #transaction costs
directory = "Results"

if not os.path.exists(directory):
    os.makedirs(directory)

msg = "" 
address = symbol + '.csv'
  

try:
    dfP = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    dfP.to_csv(address, header = True, index=True, encoding='utf-8')
except Exception:
    msg = "yahoo problem"
    dfP = pd.DataFrame()

dfP = pd.read_csv(address, parse_dates=['Date'])
dfP = dfP.sort_values(by='Date')
dfP.set_index('Date', inplace = True)

dfT, dfP = np.split(dfP, [int(.6*len(dfP))])


  
#dfP['Adj Close'].plot()
#plt.show()

dfP = dfP.assign(time = pd.Series(np.arange(dfP.shape[0])).values) 

#rolling regression instead of moving average
a = np.array([np.nan] * len(dfP))
b = [np.nan] * len(dfP)  # If betas required.
y_ = dfP['Adj Close'].values
x_ = dfP[['time']].assign(constant=1).values
for n in range(window, len(dfP)):
    y = y_[(n - window):n]
    X = x_[(n - window):n]
    # betas = Inverse(X'.X).X'.y
    betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    y_hat = betas.dot(x_[n, :])
    a[n] = y_hat
    b[n] = betas.tolist()  # If betas required.

dfP = dfP.assign(y_hat = pd.Series(a).values)
    
if regression == 1:
    mean = dfP['y_hat']
else:
    mean = dfP['Adj Close'].rolling(window=window).mean()

std_dev = dfP['Adj Close'].rolling(window=window).std()
dfP['zScore'] = (dfP['Adj Close']-mean)/std_dev
dfP['mean'] = mean
dfP['stdev'] = std_dev
dfP['LB'] = mean - entryZscore*std_dev
dfP['UB'] = mean + entryZscore*std_dev


#dfP['zScore'].plot()
#plt.show()

#set up num_units_long  
if complex_entrance == 1:           
    dfP['long_entry'] = ((dfP.zScore > - entryZscore) & ( dfP.zScore.shift(1) < - entryZscore)) 
    dfP['long_exit'] = ((dfP.zScore < - exitZscore) & (dfP.zScore.shift(1) > - exitZscore)) 
else:
    dfP['long_entry'] = ((dfP.zScore <  -entryZscore))
    dfP['long_exit'] = ((dfP.zScore >  -exitZscore))


dfP = dfP.assign(long_forbidden = pd.Series(np.zeros(dfP.shape[0])).values)
dfP.loc[(dfP.zScore < - (entryZscore+1)) & (dfP.zScore.shift(1) > - (entryZscore+1)),'long_forbidden'] = 1 


dfP['num_units_long'] = np.nan #it is important to start with nan in this column otherwise padding wont work
dfP.loc[dfP['long_entry'],'num_units_long'] = 1 
dfP.loc[dfP['long_exit'],'num_units_long'] = 0 
if forbidden == 1:
    dfP.loc[dfP['long_forbidden']==1, 'num_units_long']= 0   
dfP.iat[0,dfP.columns.get_loc("num_units_long")]= 0
dfP['num_units_long'] = dfP['num_units_long'].fillna(method='pad') 

#set up num units short 
if complex_entrance == 1:
    dfP['short_entry'] = ((dfP.zScore <  entryZscore) & ( dfP.zScore.shift(1) > entryZscore))
    dfP['short_exit'] = ((dfP.zScore > exitZscore) & (dfP.zScore.shift(1) < exitZscore))
else:
    dfP['short_entry'] = ((dfP.zScore >  entryZscore))
    dfP['short_exit'] = ((dfP.zScore <  exitZscore))


dfP = dfP.assign(short_forbidden = pd.Series(np.zeros(dfP.shape[0])).values)
dfP.loc[(dfP.zScore >  entryZscore+1) & (dfP.zScore.shift(1) <  entryZscore+1),'short_forbidden'] = 1 

dfP['num_units_short'] = np.nan 
dfP.loc[dfP['short_entry'],'num_units_short'] = -1 
dfP.loc[dfP['short_exit'],'num_units_short'] = 0
if forbidden == 1:
    dfP.loc[dfP['short_forbidden']==1, 'num_units_short']= 0 
dfP['num_units_short'] = dfP['num_units_short'].fillna(method='pad')
dfP['num_units'] = dfP['num_units_long']*(1-tcost) +  dfP['num_units_short']*shorts*(1-tcost)

#log return calculation & cummulative return
#dfP['log_rets'] = np.log(dfP['Adj Close']/dfP['Adj Close'].shift(1))
#dfP['port_rets'] = dfP['log_rets'] * dfP['num_units'].shift(2) 
#dfP['cum_rets'] = dfP['log_rets'].cumsum()
#dfP['cum_rets'] = dfP['cum_rets'] + 1

#pct return calculation & cummulative return same result as above
dfP['pct_ch'] = (dfP['Adj Close']-dfP['Adj Close'].shift(1))/abs(dfP['Adj Close'].shift(1)) 
dfP['port_rets'] = dfP['pct_ch'] * dfP['num_units'].shift(delay) 
dfPI = pd.DataFrame(data=dfP['port_rets'], index=dfP.index)
dfPI = dfPI.fillna(value=0)
dfPI = dfPI.assign(I = pd.Series(np.zeros(dfPI.shape[0])).values) 

start = 1
dfPI.iat[0,1] = start
rows = dfPI.shape[0]
for row in range(1,rows):
    dfPI.iat[row,1] = dfPI.iat[row-1,1]*dfPI.iat[row,0]+dfPI.iat[row-1,1]

dfP = dfP.assign(I = dfPI['I'])

title=symbol
plt.plot(dfP['I'])
plt.ylabel(symbol)
plt.show()
#plt.savefig(r'Results\%s.png' %(title))


start_val = start
end_val = dfP['I'].iat[-1]

start_date = dfP.iloc[0].name
end_date = dfP.iloc[-1].name
days = (end_date - start_date).days    

TotaAnnReturn = (end_val-start_val)/start_val/(days/360)
TotaAnnReturn_trading = (end_val-start_val)/start_val/(days/252)

CAGRdbl = round(((float(end_val) / float(start_val)) ** (1/(days/252.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part

try:
    sharpe =  TotaAnnReturn_trading/( (dfP['port_rets'].std()) * sqrt(252))
except ZeroDivisionError:
    sharpe = 0.0

print ("TotaAnnReturn = %f" %(TotaAnnReturn*100))
print ("CAGR = %f" %(CAGRdbl*100))
print ("Sharpe Ratio = %f" %(round(sharpe,2)))


    
dfP.to_csv(r'Results\dfP.csv', header = True, index=True, encoding='utf-8')




