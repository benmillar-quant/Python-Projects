import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

# Read CSV file into a DataFrame
data = pd.read_csv("xxDow_shortfiltered_data_2020_2025_3mV2.csv")

# Convert 'Datetime' column to datetime type
data['Datetime'] = pd.to_datetime(data['Datetime']) # , format='%Y-%m-%d %H:%M:%S'

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
data['Date'] = pd.to_datetime(data['Date'])   # , format='%d/%m/%Y'

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

### CALCULATING THE RSI ###
overbought_level = 50.5
oversold_level = 49.49
rsiw = 21
# Use Datetime date for grouping
data['Date'] = pd.to_datetime(data['Datetime']).dt.date
if not data.index.is_unique:
    data.reset_index(drop=True, inplace=True)
# Initialize RSI helper columns
for col in ['Change','Gain','Loss','AvgGain','AvgLoss','RS','RSI_21']:
    data[col] = np.nan

for day in data['Date'].unique():
    idx = data[data['Date'] == day].index
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
                dd.loc[idx[i],'AvgGain'] = ((dd.loc[idx[i-1],'AvgGain']*(rsiw-1) + dd.loc[idx[i],'Gain'])/rsiw)
                dd.loc[idx[i],'AvgLoss'] = ((dd.loc[idx[i-1],'AvgLoss']*(rsiw-1) + dd.loc[idx[i],'Loss'])/rsiw)
            dd.loc[idx[i],'RS']     = dd.loc[idx[i],'AvgGain'] / dd.loc[idx[i],'AvgLoss']
            dd.loc[idx[i],'RSI_21'] = 100 - (100/(1+dd.loc[idx[i],'RS']))
        data.update(dd)

data.drop(columns=['Date'], inplace=True)

### PLOT RSI AND ATR CHARTS ###
# RSI
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Datetime'], y=data['RSI_21'], mode='lines', name='RSI_21'))
fig.add_trace(go.Scatter(x=data['Datetime'], y=[overbought_level]*len(data), mode='lines', name='Overbought', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=data['Datetime'], y=[oversold_level]*len(data), mode='lines', name='Oversold', line=dict(dash='dash')))
fig.update_layout(title='RSI Chart', xaxis_title='Date and Time', yaxis_title='RSI')
# fig.show()

# ATR
atr_period = 5
data['range'] = data['High'] - data['Low']
data['ATR']   = data['range'].rolling(atr_period).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Datetime'], y=data['ATR'], mode='lines', name='ATR'))
fig.update_layout(title='ATR Chart', xaxis_title='Date and Time', yaxis_title='ATR')
# fig.show()

### CALCULATING THE BACKTESTING LOGIC ###
# Creating columns
cols = ['status','entry_price','stop_loss','exit_price','exit_time',
        'take_profit_target','risk','outcome','daily_losses','total_losses','total_wins', 'PnL']
for c in cols:
    if c in ['status','exit_time','outcome','daily_losses','total_losses','total_wins']:
        data[c] = None
    elif c == 'PnL':
        data[c] = 0.0
    else:
        data[c] = np.nan

# Initialize variables
trade_open = False
daily_losses = 0
total_losses = 0
total_wins = 0
cumulative_pnl = 0
previous_date = None
last_daily_atr = np.nan
stop_loss_moved_to_be = False
below_oversold = False

# Strategy params
starting_balance = 100000
TPTarget = 80
per_pt = 20
max_risk = 80
spread = 3
daily_atr_divider = 2
intraday_atr_multiplier = 1.25
eod = "14:57:00"
stop_loss_to_BE = 30
atr_stop_loss_multiplier = 2

active_trades = []

for i in range(len(data)):
    current_date = data['Datetime'].iloc[i].date()
    try:
        current_time = str(data['Datetime'].iloc[i]).split()[1]
    except:
        continue

    # New day
    if current_date != previous_date:
        trade_open_hours = [("07:30:00", "08:30:00"), ("08:30:00", "14:57:00")]
        daily_losses = 0
        for t in active_trades:
            if t.get('active', False): t['active'] = False
        active_trades = []
        previous_date = current_date

    # Determine whether we’re in the allowed opening window
    is_open_time = any(st <= current_time <= et for st,et in trade_open_hours)

    # === OPENING TRADES (only during open hours & under daily loss limit) ===
    if is_open_time and daily_losses < 3:
        # Single-trade check
        can_open_new_trade = not any(t['active'] for t in active_trades)
        for t in active_trades:
            if t['active'] and not t['stop_loss_moved_to_be']:
                can_open_new_trade = False
                break

        if can_open_new_trade:
            if data['RSI_21'].iloc[i] <= oversold_level:
                below_oversold = True

            # 2) wait for RSI to cross into overbought
            if (
                below_oversold
                and data['RSI_21'].iloc[i-1] <= overbought_level
                and data['RSI_21'].iloc[i]   >= overbought_level
            ):

                # look at bar i+1, then i+2, then i+3 — in each: breakout first, then cancellation
                for j in range(i+1, i+4):

                    # --- (1) breakout check on this bar ---
                    if data['High'].iloc[j] >= data['High'].iloc[i] + spread:
                        risk_value = (data['High'].iloc[i] + spread) - (data['Low'].iloc[i] - spread)
                        if risk_value <= max_risk:
                            t = {
                                'active': True,
                                'status': 'Buy Order Opened',
                                'entry_price': data['High'].iloc[i] + spread,
                                'stop_loss':  data['Low'].iloc[i] - spread,
                                'risk':       risk_value,
                                'take_profit_target': data['High'].iloc[i] + spread + (last_daily_atr/daily_atr_divider),
                                'trade_open_idx':     j,
                                'stop_loss_moved_to_be': False
                            }
                            active_trades.append(t)

                            data.at[j, 'status']              = t['status']
                            data.at[j, 'entry_price']         = t['entry_price']
                            data.at[j, 'stop_loss']           = t['stop_loss']
                            data.at[j, 'risk']                = t['risk']
                            data.at[j, 'take_profit_target']  = t['take_profit_target']

                            # reset and exit loop once trade opens
                            below_oversold = False
                            break

                    # --- (2) cancellation check on this bar ---
                    if data['RSI_21'].iloc[j] <= oversold_level:
                        # RSI dipped back into oversold *before* any breakout → abort this signal
                        below_oversold = True
                        break

    # === MANAGING OPEN TRADES (always, regardless of time) ===
    for t in active_trades:
        if not t['active'] or i <= t['trade_open_idx']:
            continue

        # 1) Stop-loss
        if data['Low'].iloc[i] <= t['stop_loss']:
            t['active'] = False
            t['stop_loss_moved_to_be'] = False
            t['exit_time'] = data['Time'].iloc[i]
            exit_price = t['stop_loss']
            t['exit_price'] = exit_price
            pnl = exit_price - t['entry_price']
            cumulative_pnl += pnl
            if pnl < 0:
                outcome = 'Loss'
                daily_losses += 1
                total_losses += 1
            elif pnl == 0:
                outcome = 'B/E'
            data.at[t['trade_open_idx'],'entry_price'] = t['entry_price']
            data.at[t['trade_open_idx'],'exit_price'] = t['exit_price']
            data.at[t['trade_open_idx'],'exit_time'] = t['exit_time']
            data.at[t['trade_open_idx'],'stop_loss'] = t['stop_loss']
            data.at[t['trade_open_idx'],'risk'] = t['risk']
            data.at[t['trade_open_idx'],'outcome'] = outcome
            data.at[t['trade_open_idx'],'PnL'] = pnl
            data.at[t['trade_open_idx'],'daily_losses'] = daily_losses
            data.at[t['trade_open_idx'],'total_wins'] = total_wins
            data.at[t['trade_open_idx'],'total_losses'] = total_losses

        # # # # 2) Move SL to BE
        if not t['stop_loss_moved_to_be'] and data['High'].iloc[i] >= t['entry_price'] + stop_loss_to_BE:
            t['stop_loss'] = t['entry_price']
            t['stop_loss_moved_to_be'] = True
            data.at[t['trade_open_idx'],'stop_loss'] = t['entry_price']

        # 5) RSI-based TP
        if data['RSI_21'].iloc[i] <= oversold_level:
            tp_met = False
            for j in range(i+1, len(data)):
                if data['RSI_21'].iloc[j] >= overbought_level:
                    break
                if data['Low'].iloc[j] <= data['Low'].iloc[i] - spread:
                    pot = (data['Low'].iloc[i] - spread) - t['entry_price']
                    if pot >= 1:
                        tp_met = True
                        t['exit_price'] = data['Low'].iloc[i] - spread
                        break
            if tp_met:
                t['active'] = False
                t['stop_loss_moved_to_be'] = False
                pnl = t['exit_price'] - t['entry_price']
                cumulative_pnl += pnl
                t['outcome'] = 'Win'
                t['exit_time'] = data['Time'].iloc[i]
                total_wins += 1
                data.at[t['trade_open_idx'],'entry_price'] = t['entry_price']
                data.at[t['trade_open_idx'],'exit_price'] = t['exit_price']
                data.at[t['trade_open_idx'],'exit_time'] = t['exit_time']
                data.at[t['trade_open_idx'],'stop_loss'] = t['stop_loss']
                data.at[t['trade_open_idx'],'risk'] = t['risk']
                data.at[t['trade_open_idx'],'outcome'] = t['outcome']
                data.at[t['trade_open_idx'],'PnL'] = pnl
                data.at[t['trade_open_idx'],'daily_losses'] = daily_losses
                data.at[t['trade_open_idx'],'total_wins'] = total_wins
                data.at[t['trade_open_idx'],'total_losses'] = total_losses

        # 6) EOD close
        if data['Time'].iloc[i] == eod:
            exit_price = data['Close'].iloc[i]
            t['exit_price'] = exit_price
            pnl = exit_price - t['entry_price']
            cumulative_pnl += pnl
            t['exit_time'] = data['Time'].iloc[i]
            t['active'] = False
            t['stop_loss_moved_to_be'] = False
            if pnl > 0:
                outcome = 'Win'
                total_wins += 1
            elif pnl < 0:
                outcome = 'Loss'
                daily_losses += 1
                total_losses += 1
            else:
                outcome = 'B/E'
            data.at[t['trade_open_idx'],'entry_price'] = t['entry_price']
            data.at[t['trade_open_idx'],'exit_price'] = t['exit_price']
            data.at[t['trade_open_idx'],'exit_time'] = t['exit_time']
            data.at[t['trade_open_idx'],'stop_loss'] = t['stop_loss']
            data.at[t['trade_open_idx'],'risk'] = t['risk']
            data.at[t['trade_open_idx'],'outcome'] = outcome
            data.at[t['trade_open_idx'],'PnL'] = pnl
            data.at[t['trade_open_idx'],'daily_losses'] = daily_losses
            data.at[t['trade_open_idx'],'total_wins'] = total_wins
            data.at[t['trade_open_idx'],'total_losses'] = total_losses
            active_trades = []
            continue

### CALCULATING PNL AND STRATEGY RESULTS ###
# Cumulative PnL
data['cumulative_PnL'] = data['PnL'].cumsum()
plt.figure(figsize=(10,5))
plt.plot(data['Datetime'], data['cumulative_PnL'], label='Cumulative PnL')
plt.title('Cumulative Profit and Loss')
plt.xlabel('Datetime')
plt.ylabel('Cumulative PnL')
plt.grid(True)
plt.legend()
plt.show()

print("--RSI BUY--")
print("Total Wins:", total_wins)
print("Total Losses:", total_losses)
if total_wins + total_losses > 0:
    print(f"Win/Loss Percentage: {(total_wins/(total_wins+total_losses))*100:.2f}%")
else:
    print("No completed trades to calculate win/loss percentage.")
print(f"Biggest Win: {data['PnL'].max():.2f} Pts")
print(f"Biggest Loss: {data['PnL'].min():.2f} Pts")
print(f"Cumulative PnL: {cumulative_pnl:.2f} Pts")
print(f"Starting balance: £{starting_balance}")
print(f"Cash returns @ £{per_pt} per pt: £{cumulative_pnl*per_pt:.2f}")

cash_returns = cumulative_pnl*per_pt
final_balance = starting_balance + cash_returns
print(f"Percentage returns: {((final_balance-starting_balance)/starting_balance)*100:.2f}%")

# === CALCULATE DRAWDOWN FROM CUMULATIVE PnL ===
data['peak'] = data['cumulative_PnL'].cummax()
data['DD'] = data['cumulative_PnL'] - data['peak']
data['DD%'] = data['DD'] / data['peak'].replace(0, np.nan) * 100

# Capture max drawdown and peak
max_drawdown_points = data['DD'].min()
max_drawdown_percent = data['DD%'].min()
peak_points = data['peak'].max()

print(f"Peak PnL: {peak_points:.2f} pts")
print(f"Maximum Drawdown: {max_drawdown_points:.2f} pts")
print(f"Maximum Drawdown Percentage: {max_drawdown_percent:.2f}%")

# Cleanup and export
cols_to_drop = ['Change','Gain','Loss','AvgGain','AvgLoss','RS','range','ATR', 'Daily_ATR', 'take_profit_target']
data.drop(columns=cols_to_drop, inplace=True)
data.to_excel("RSI_Results_Buy.xlsx")
