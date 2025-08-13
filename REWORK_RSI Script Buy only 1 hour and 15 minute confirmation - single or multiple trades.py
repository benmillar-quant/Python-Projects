import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

# Read CSV file into a DataFrame
data = pd.read_csv("xxDow_longfiltered_data_2020_2025_3mV2.csv")

# Convert 'Datetime' column to datetime type
data['Datetime'] = pd.to_datetime(data['Datetime'])

### DISPLAYING THE PRICE BAR CHART ###
fig_price = go.Figure(data=[go.Candlestick(x=data['Datetime'],
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'])])
fig_price.update_layout(title='DJIA - Price Bar Chart',
                        xaxis_title='Datetime',
                        yaxis_title='Price',
                        xaxis_rangeslider_visible=False)
# fig_price.show()

# Prepare 'Date' for daily ATR
data['Date'] = pd.to_datetime(data['Date'])

### CALCULATING THE DAILY ATR ###
daily_high_low = data.groupby(data['Date'].dt.date).agg(
    Daily_High=('High', 'max'),
    Daily_Low=('Low', 'min')
).reset_index()
daily_high_low['Date'] = pd.to_datetime(daily_high_low['Date'])
daily_high_low['Daily_Range'] = daily_high_low['Daily_High'] - daily_high_low['Daily_Low']
daily_high_low['Daily_ATR'] = daily_high_low['Daily_Range'].rolling(window=5).mean()
data = pd.merge(data, daily_high_low[['Date', 'Daily_ATR']], on='Date', how='left')
data['Daily_ATR'] = data.groupby(data['Date'].dt.date)['Daily_ATR'].shift(1)

### RSI SETTINGS ###
overbought_level = 50.5
oversold_level   = 49.49
rsiw = 21

# Ensure unique index
if not data.index.is_unique:
    data.reset_index(drop=True, inplace=True)

# --- 1) Calculate 3-minute RSI_21 ---
for col in ['Change','Gain','Loss','AvgGain','AvgLoss','RS','RSI_21']:
    data[col] = np.nan
for day in data['Datetime'].dt.date.unique():
    idx = data[data['Datetime'].dt.date == day].index
    if len(idx) > rsiw:
        dd = data.loc[idx].copy()
        dd['Change'] = dd['Close'].diff()
        dd['Gain']   = dd['Change'].apply(lambda x: x if x>0 else 0)
        dd['Loss']   = dd['Change'].apply(lambda x: abs(x) if x<0 else 0)
        for i in range(rsiw, len(idx)):
            if i == rsiw:
                dd.loc[idx[i],'AvgGain'] = dd['Gain'].iloc[1:rsiw+1].mean()
                dd.loc[idx[i],'AvgLoss'] = dd['Loss'].iloc[1:rsiw+1].mean()
            else:
                prev = idx[i-1]
                curr = idx[i]
                dd.loc[curr,'AvgGain'] = (dd.loc[prev,'AvgGain']*(rsiw-1) + dd.loc[curr,'Gain'])/rsiw
                dd.loc[curr,'AvgLoss'] = (dd.loc[prev,'AvgLoss']*(rsiw-1) + dd.loc[curr,'Loss'])/rsiw
            dd.loc[idx[i],'RS']     = dd.loc[idx[i],'AvgGain'] / dd.loc[idx[i],'AvgLoss']
            dd.loc[idx[i],'RSI_21'] = 100 - (100/(1+dd.loc[idx[i],'RS']))
        data.update(dd)

data.drop(columns=['Date'], inplace=True)

# --- 2) RSI helper: function for resampled series ---
def compute_rsi_series(close_series, period):
    df = pd.DataFrame({'Close': close_series})
    df['Change'] = df['Close'].diff()
    df['Gain']   = df['Change'].apply(lambda x: x if x>0 else 0)
    df['Loss']   = df['Change'].apply(lambda x: abs(x) if x<0 else 0)
    df['AvgGain'] = np.nan
    df['AvgLoss'] = np.nan
    for i in range(period, len(df)):
        if i == period:
            df.loc[df.index[i], 'AvgGain'] = df['Gain'].iloc[1:period+1].mean()
            df.loc[df.index[i], 'AvgLoss'] = df['Loss'].iloc[1:period+1].mean()
        else:
            prev = df.index[i-1]
            curr = df.index[i]
            df.loc[curr, 'AvgGain'] = (df.loc[prev, 'AvgGain']*(period-1) + df.loc[curr, 'Gain'])/period
            df.loc[curr, 'AvgLoss'] = (df.loc[prev, 'AvgLoss']*(period-1) + df.loc[curr, 'Loss'])/period
    df['RS'] = df['AvgGain'] / df['AvgLoss']
    df['RSI'] = 100 - (100/(1+df['RS']))
    return df['RSI']

# Resample to 15-minute and 1-hour closes
resampled = data.set_index('Datetime')
close_15 = resampled['Close'].resample('15T').last().dropna()
close_60 = resampled['Close'].resample('60T').last().dropna()

# Compute multi-timeframe RSI
rsi_15 = compute_rsi_series(close_15, rsiw)
rsi_15.name = 'RSI_15'
rsi_60 = compute_rsi_series(close_60, rsiw)
rsi_60.name = 'RSI_60'

# Map back to each 3m bar via period-end timestamp
data['PeriodEnd_15'] = data['Datetime'].dt.floor('15T')
data['PeriodEnd_60'] = data['Datetime'].dt.floor('60T')
data['RSI_15'] = data['PeriodEnd_15'].map(rsi_15)
data['RSI_60'] = data['PeriodEnd_60'].map(rsi_60)
data.drop(columns=['PeriodEnd_15','PeriodEnd_60'], inplace=True)

### PLOT COMBINED RSI ###
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Datetime'], y=data['RSI_21'], mode='lines', name='RSI 3m'))
fig.add_trace(go.Scatter(x=data['Datetime'], y=data['RSI_15'], mode='lines', name='RSI 15m'))
fig.add_trace(go.Scatter(x=data['Datetime'], y=data['RSI_60'], mode='lines', name='RSI 1h'))
fig.add_trace(go.Scatter(x=data['Datetime'], y=[overbought_level]*len(data),
                         mode='lines', name='Threshold', line=dict(dash='dash')))
fig.update_layout(title='RSI Across Timeframes', xaxis_title='Datetime', yaxis_title='RSI')
# fig.show()

### CALCULATE ATR ###
data['range'] = data['High'] - data['Low']
data['ATR']   = data['range'].rolling(5).mean()

### BACKTESTING LOGIC ###
cols = ['status','entry_price','stop_loss','exit_price','exit_time',
        'take_profit_target','risk','outcome','daily_losses','total_losses','total_wins','PnL']
for c in cols:
    if c in ['status','exit_time','outcome','daily_losses','total_losses','total_wins']:
        data[c] = None
    elif c=='PnL':
        data[c] = 0.0
    else:
        data[c] = np.nan

# Variables
daily_losses = total_losses = total_wins = 0
cumulative_pnl = 0
previous_date = None
active_trades = []

# Params
eod = "14:57:00"
spread = 3
max_risk = 80
stop_loss_to_BE = 30
daily_atr_divider = 2

for i in range(len(data)):
    dt = data['Datetime'].iloc[i]
    current_date = dt.date()
    current_time = dt.time().strftime('%H:%M:%S')

    # New trading day reset
    if current_date != previous_date:
        trade_hours = [("07:30:00","08:30:00"),("08:30:00","14:57:00")]
        daily_losses = 0
        for t in active_trades:
            t['active'] = False
        active_trades = []
        previous_date = current_date

    # Within open window?
    is_open = any(start <= current_time <= end for start,end in trade_hours)

    # === OPEN LOGIC ===
    if is_open and daily_losses<3:
        # Only one active
        if any(t['active'] and not t['stop_loss_moved_to_be'] for t in active_trades):
            pass
        else:
            # Multi-timeframe filter
            if pd.notna(data['RSI_15'].iloc[i]) and data['RSI_15'].iloc[i]>overbought_level \
               and pd.notna(data['RSI_60'].iloc[i]) and data['RSI_60'].iloc[i]>overbought_level:
                # 3m oversold-entry
                if data['RSI_21'].iloc[i] <= oversold_level:
                    below_oversold = True
                if below_oversold and data['RSI_21'].iloc[i-1]<=overbought_level and data['RSI_21'].iloc[i]>=overbought_level:
                    # look ahead bars
                    for j in range(i+1, min(len(data), i+4)):
                        # breakout
                        if data['High'].iloc[j]>=data['High'].iloc[i]+spread:
                            risk = (data['High'].iloc[i]+spread)-(data['Low'].iloc[i]-spread)
                            if risk<=max_risk:
                                t = {'active':True,'status':'Buy Order Opened',
                                     'entry_price':data['High'].iloc[i]+spread,
                                     'stop_loss':data['Low'].iloc[i]-spread,
                                     'risk':risk,
                                     'take_profit_target':data['High'].iloc[i]+spread+(data['ATR'].iloc[i]/daily_atr_divider),
                                     'trade_open_idx':j,'stop_loss_moved_to_be':False}
                                active_trades.append(t)
                                data.at[j,'status']=t['status']; data.at[j,'entry_price']=t['entry_price']
                                data.at[j,'stop_loss']=t['stop_loss']; data.at[j,'risk']=t['risk']
                                data.at[j,'take_profit_target']=t['take_profit_target']
                                below_oversold=False
                                break
                        # cancel if RSI dips back
                        if data['RSI_21'].iloc[j] <= oversold_level:
                            below_oversold=True; break

    # === MANAGEMENT ===
    for t in active_trades:
        if not t['active'] or i<=t['trade_open_idx']: continue
        low = data['Low'].iloc[i]; high = data['High'].iloc[i]
        # SL hit
        if low <= t['stop_loss']:
            exit_p = t['stop_loss']; pnl = exit_p - t['entry_price']; t['active']=False
            daily_losses += pnl<0; total_losses += pnl<0; cumulative_pnl+=pnl
            data.at[t['trade_open_idx'],'exit_price']=exit_p; data.at[t['trade_open_idx'],'PnL']=pnl
        # move SL to BE
        if not t['stop_loss_moved_to_be'] and high>=t['entry_price']+stop_loss_to_BE:
            t['stop_loss']=t['entry_price']; t['stop_loss_moved_to_be']=True
            data.at[t['trade_open_idx'],'stop_loss']=t['stop_loss']
        # RSI-based TP
        if data['RSI_21'].iloc[i] <= oversold_level:
            for j in range(i+1,len(data)):
                if data['RSI_21'].iloc[j]>=overbought_level: break
                if data['Low'].iloc[j] <= data['Low'].iloc[i]-spread:
                    t['exit_price'] = data['Low'].iloc[i]-spread; t['active']=False; pnl = t['exit_price']-t['entry_price']
                    cumulative_pnl+=pnl; total_wins+=1
                    data.at[t['trade_open_idx'],'exit_price']=t['exit_price']; data.at[t['trade_open_idx'],'PnL']=pnl
                    break
        # EOD close
        if current_time==eod and t['active']:
            exit_p = data['Close'].iloc[i]; pnl = exit_p - t['entry_price'];
            t['active']=False; cumulative_pnl+=pnl; daily_losses += pnl<0; total_losses+=pnl<0; total_wins+=pnl>0
            data.at[t['trade_open_idx'],'exit_price']=exit_p; data.at[t['trade_open_idx'],'PnL']=pnl

# === POST PROCESS & RESULTS ===
data['cumulative_PnL'] = data['PnL'].cumsum()
plt.figure(figsize=(10,5)); plt.plot(data['Datetime'], data['cumulative_PnL']); plt.title('Cumulative PnL'); plt.show()
print("Total Wins:", total_wins, "Total Losses:", total_losses)
print(f"Cumulative PnL: {cumulative_pnl:.2f} pts")

# Cleanup & export
drop_cols=['Change','Gain','Loss','AvgGain','AvgLoss','RS','range']
data.drop(columns=drop_cols, inplace=True)
data.to_excel("RSI_Results_Buy_Updated.xlsx", index=False)
