import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import os


# Read CSV file into a DataFrame
data = pd.read_csv("DJIA_DAILY_DATA_SEP_TO_DEC.csv")

# Convert 'Datetime' column to datetime type
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d/%m/%Y %H:%M')    #, format='%d/%m/%Y %H:%M')

### DISPLAYING THE PRICE BAR CHART ###

# Create the price bar chart
fig_price = go.Figure(data=[go.Candlestick(x=data['Datetime'],
                                           open=data['Open'],
                                           high=data['High'],
                                           low=data['Low'],
                                           close=data['Close'])])

# Set the layout for the price chart
fig_price.update_layout(title='DJIA - Price Bar Chart',
                        xaxis_title='Datetime',
                        yaxis_title='Price',
                        xaxis_rangeslider_visible=False)

# Display the price chart
fig_price.show()

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')  #, format='%d/%m/%Y')

### CALCULATING THE DAILY ATR ###

# Step 1: Calculate daily highest high and lowest low
daily_high_low = data.groupby(data['Date'].dt.date).agg(Daily_High=('High', 'max'), Daily_Low=('Low', 'min')).reset_index()
daily_high_low['Date'] = pd.to_datetime(daily_high_low['Date'])
daily_high_low['Daily_Range'] = daily_high_low['Daily_High'] - daily_high_low['Daily_Low']

# Step 2: Calculate the 5-day average of these Daily Ranges
daily_high_low['Daily_ATR'] = daily_high_low['Daily_Range'].rolling(window=5).mean()

# Step 3: Merge this information back to the original DataFrame to get the date alignment
data = pd.merge(data, daily_high_low[['Date', 'Daily_ATR']], on='Date', how='left')

# Step 4: Shift the Daily ATR value down to apply it to the first row of the next day
data['Daily_ATR'] = data.groupby(data['Date'].dt.date)['Daily_ATR'].shift(1)

### CALCULATING THE RSI ###

# RSI Overbought & Oversold levels
overbought_level = 50.5
oversold_level = 49.49

# RSI period
rsiw = 21

data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')    #, format='%d/%m/%Y')
if not data.index.is_unique:
    data.reset_index(drop=True, inplace=True)

# Creating columns
data['Change'] = np.nan
data['Gain'] = np.nan
data['Loss'] = np.nan
data['AvgGain'] = np.nan
data['AvgLoss'] = np.nan
data['RS'] = np.nan
data['RSI_21'] = np.nan

data['Date'] = pd.to_datetime(data['Datetime']).dt.date

# Process each day separately so that RSI calculations start fresh when a new day is detected in the data
for day in data['Date'].unique():
    day_indices = data[data['Date'] == day].index
    if len(day_indices) > rsiw:

        day_data = data.loc[day_indices].copy()
        
        # Calculate Change, Gain, Loss
        day_data['Change'] = day_data['Close'].diff()
        day_data['Gain'] = day_data['Change'].apply(lambda x: x if x > 0 else 0)
        day_data['Loss'] = day_data['Change'].apply(lambda x: abs(x) if x < 0 else 0)

        # Calculate AvgGain, AvgLoss, RS, and RSI starting from the 21st record
        for i in range(rsiw, len(day_indices)):
            if i == rsiw:
                day_data.loc[day_indices[i], 'AvgGain'] = day_data['Gain'].iloc[1:rsiw+1].mean()
                day_data.loc[day_indices[i], 'AvgLoss'] = day_data['Loss'].iloc[1:rsiw+1].mean()
            else:
                day_data.loc[day_indices[i], 'AvgGain'] = (day_data.loc[day_indices[i-1], 'AvgGain'] * (rsiw - 1) + day_data.loc[day_indices[i], 'Gain']) / rsiw
                day_data.loc[day_indices[i], 'AvgLoss'] = (day_data.loc[day_indices[i-1], 'AvgLoss'] * (rsiw - 1) + day_data.loc[day_indices[i], 'Loss']) / rsiw
            
            day_data.loc[day_indices[i], 'RS'] = day_data.loc[day_indices[i], 'AvgGain'] / day_data.loc[day_indices[i], 'AvgLoss']
            day_data.loc[day_indices[i], 'RSI_21'] = 100 - (100 / (1 + day_data.loc[day_indices[i], 'RS']))
        
        # Update the original DataFrame with calculated values
        data.update(day_data)

data.drop(columns=['Date'], inplace=True)

### PLOT RSI VALUES AND DISPLAY ON CHART ###
fig = go.Figure()

fig.add_trace(go.Scatter(x=data['Datetime'], y=data['RSI_21'], mode='lines', name='RSI_21'))
fig.add_trace(go.Scatter(x=data['Datetime'], y=[overbought_level] * len(data['RSI_21']), mode='lines', name='Overbought', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=data['Datetime'], y=[oversold_level] * len(data['RSI_21']), mode='lines', name='Oversold', line=dict(dash='dash')))

fig.update_layout(title='RSI Chart', xaxis_title='Date and Time', yaxis_title='RSI')

# Display chart
fig.show()

### CALCULATE INTRADAY PERIOD ATR AND DISPLAY VALUES ON CHART ###

atr_period = 5

# calculating the range
data['range'] = data['High'] - data['Low']

# calculating the average value of ranges
data['ATR'] = data['range'].rolling(atr_period).mean()

# Plot ATR values and display chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Datetime'], y=data['ATR'], mode='lines', name='ATR'))
fig.update_layout(title='Average True Range (ATR) Chart', xaxis_title='Date and Time', yaxis_title='ATR')
# fig.show()

### CALCULATING THE DAILY RSI ###
# Calculate daily OHLC
daily_data = data.groupby(data['Datetime'].dt.date).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).reset_index()
daily_data.rename(columns={'Datetime': 'Date'}, inplace=True)

# Creating columns for daily RSI calculation
daily_data['Change'] = daily_data['Close'].diff()
daily_data['Gain'] = daily_data['Change'].apply(lambda x: x if x > 0 else 0)
daily_data['Loss'] = daily_data['Change'].apply(lambda x: abs(x) if x < 0 else 0)
daily_data['AvgGain'] = np.nan
daily_data['AvgLoss'] = np.nan
daily_data['RS'] = np.nan
daily_data['RSI_21'] = np.nan

# Calculate AvgGain, AvgLoss, RS, and RSI
for i in range(rsiw, len(daily_data)):
    if i == rsiw:
        daily_data.loc[i, 'AvgGain'] = daily_data['Gain'].iloc[1:rsiw+1].mean()
        daily_data.loc[i, 'AvgLoss'] = daily_data['Loss'].iloc[1:rsiw+1].mean()
    else:
        daily_data.loc[i, 'AvgGain'] = (daily_data.loc[i-1, 'AvgGain'] * (rsiw - 1) + daily_data.loc[i, 'Gain']) / rsiw
        daily_data.loc[i, 'AvgLoss'] = (daily_data.loc[i-1, 'AvgLoss'] * (rsiw - 1) + daily_data.loc[i, 'Loss']) / rsiw

    daily_data.loc[i, 'RS'] = daily_data.loc[i, 'AvgGain'] / daily_data.loc[i, 'AvgLoss']
    daily_data.loc[i, 'RSI_21'] = 100 - (100 / (1 + daily_data.loc[i, 'RS']))

# Shift daily RSI to align with the next day
daily_data['RSI_21'] = daily_data['RSI_21'].shift(1)

# Create a daily candlestick chart using the calculated daily OHLC data
fig_daily = go.Figure(data=[go.Candlestick(x=daily_data['Date'],
                                           open=daily_data['Open'],
                                           high=daily_data['High'],
                                           low=daily_data['Low'],
                                           close=daily_data['Close'])])

# Add background shading for continuous periods with RSI_21 >= overbought_level or <= oversold_level
overbought_periods_daily = daily_data['RSI_21'] >= overbought_level
oversold_periods_daily = daily_data['RSI_21'] <= oversold_level

# Initialize variables to track the start of shading periods
start_overbought_daily = None
start_oversold_daily = None

for i in range(len(daily_data)):
    # Check for start of overbought period
    if overbought_periods_daily.iloc[i] and start_overbought_daily is None:
        start_overbought_daily = daily_data['Date'].iloc[i]
    # Check for end of overbought period
    elif not overbought_periods_daily.iloc[i] and start_overbought_daily is not None:
        fig_daily.add_shape(type='rect',
                            xref='x', yref='paper',
                            x0=start_overbought_daily, y0=0,
                            x1=daily_data['Date'].iloc[i], y1=1,
                            fillcolor='green', opacity=0.2, line_width=0)
        start_overbought_daily = None

    # Check for start of oversold period
    if oversold_periods_daily.iloc[i] and start_oversold_daily is None:
        start_oversold_daily = daily_data['Date'].iloc[i]
    # Check for end of oversold period
    elif not oversold_periods_daily.iloc[i] and start_oversold_daily is not None:
        fig_daily.add_shape(type='rect',
                            xref='x', yref='paper',
                            x0=start_oversold_daily, y0=0,
                            x1=daily_data['Date'].iloc[i], y1=1,
                            fillcolor='red', opacity=0.2, line_width=0)
        start_oversold_daily = None

# Handle the case where a period extends to the end of the data
if start_overbought_daily is not None:
    fig_daily.add_shape(type='rect',
                        xref='x', yref='paper',
                        x0=start_overbought_daily, y0=0,
                        x1=daily_data['Date'].iloc[-1], y1=1,
                        fillcolor='green', opacity=0.2, line_width=0)

if start_oversold_daily is not None:
    fig_daily.add_shape(type='rect',
                        xref='x', yref='paper',
                        x0=start_oversold_daily, y0=0,
                        x1=daily_data['Date'].iloc[-1], y1=1,
                        fillcolor='red', opacity=0.2, line_width=0)

# Set the layout for the daily chart
fig_daily.update_layout(title='DJIA - Daily Candlestick Chart',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        xaxis_rangeslider_visible=False)

# Display the daily chart
fig_daily.show()

### MERGE DAILY RSI WITH 3-MIN DATA ###
data['Date'] = data['Datetime'].dt.date
data = pd.merge(data, daily_data[['Date', 'RSI_21']], on='Date', how='left', suffixes=('', '_Daily'))

### CALCULATING THE BACKTESTING LOGIC ###

# Creating columns
data['status'] = None
data['trade_type'] = None
data['entry_price'] = np.nan
data['stop_loss'] = np.nan
data['exit_price'] = np.nan
data['exit_time'] = None
data['take_profit_target'] = np.nan
data['risk'] = np.nan
data['outcome'] = None
data['PnL'] = 0.0
data['daily_losses'] = None
data['total_losses'] = None
data['total_wins'] = None

# Initialize variables for trade tracking
trade_open = False
daily_losses = 0  # Counter for daily losses
total_losses = 0  # Counter for total losses
total_wins = 0  # Counter for total wins
cumulative_pnl = 0  # Cumulative PnL
previous_date = None
ranged = False
last_daily_atr = np.nan
stop_loss_moved_to_be = False

# Trade strategy variables
starting_balance = 100000
per_pt = 20
max_risk = 80
spread = 3
daily_atr = 375
daily_atr_divider = 2
intraday_atr_multiplier = 1.25
eod = "15:57:00"
stop_loss_to_BE = 40
atr_stop_loss_multiplier = 2

# Creating a List to hold active trades
active_trades = []

# Iterate through the DataFrame to identify buy signals and manage trades
for i in range(len(data)):
    current_date = data['Datetime'].iloc[i].date()
    try:
        current_datetime = str(data['Datetime'].iloc[i])
        current_time = current_datetime.split()[1]
    except (IndexError, AttributeError):
        continue

    # Check the previous day's daily ATR and adjust trade opening hours accordingly
    if current_date != previous_date:
        if i > 0:
            # Find the last row of the previous day (which should have the Daily ATR value)
            prev_day_last_row = data[data['Datetime'].dt.date == previous_date].index[-1]
            last_daily_atr = data.loc[prev_day_last_row, 'Daily_ATR']

        # Adjust trade opening hours based on the condition
        if last_daily_atr > daily_atr:
            trade_open_hours = [("09:30:00", "15:57:00")]
        else:
            trade_open_hours = [("09:30:00", "10:00:00"), ("10:03:00", "15:57:00")]

        # Reset daily losses counter
        daily_losses = 0

        # Close all trades at EOD
        for trade in active_trades:
            if trade.get('active', False):
                trade['active'] = False
        active_trades = []  # Reset the list for the next day
        previous_date = current_date

    # Check if the current time is within the allowed trade opening hours
    within_trade_hours = False
    for start_time, end_time in trade_open_hours:
        if start_time <= current_time <= end_time:
            within_trade_hours = True
            break

    if not within_trade_hours:
        # Skip processing if the current time is not within the allowed trade opening hours
        continue

    if daily_losses < 3:

        ### Toggle between single and multiple trades: ###
        # can_open_new_trade = not any(trade['active'] for trade in active_trades)      ##### Single trade mode #####
        can_open_new_trade = True                                                       ##### Multiple trades mode #####
        for trade in active_trades:
            if trade['active'] and not trade['stop_loss_moved_to_be']:
                can_open_new_trade = False
                break

        # Only allow a trade if the previous day's Daily RSI is >= overbought level
        if can_open_new_trade and data['RSI_21_Daily'].iloc[i] >= overbought_level:
            ### OPENING TRADES LOGIC ###

            ### TRADE TYPE A ###
            if data['RSI_21'].iloc[i-1] <= oversold_level and data['RSI_21'].iloc[i] >= overbought_level:
                # Trade A: Direct crossover from <= oversold_level to >= overbought_level
                for j in range(i+1, i+4):
                    if data['High'].iloc[j] >= data['High'].iloc[i] + spread:
                        # Check if the condition to place the trade based on risk is also satisfied
                        if (data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread)) <= max_risk:
                            # Execute trade
                            trade_details = {
                                'active': True,
                                'trade_type': 'A',
                                'status': 'Buy Trade A Opened',
                                'entry_price': data['High'].iloc[i] + spread,
                                'stop_loss': data['Low'].iloc[i] - spread,
                                'risk': data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread),
                                'take_profit_target': data['High'].iloc[i] + spread + (last_daily_atr / daily_atr_divider),
                                'trade_open_idx': j,
                                'stop_loss_moved_to_be': False
                            }
                            active_trades.append(trade_details)

                            # Update DataFrame with details of the opened trade at index j
                            data.at[j, 'trade_open'] = True
                            data.at[j, 'trade_type'] = 'A'
                            data.at[j, 'status'] = 'Buy Trade A Opened'
                            data.at[j, 'entry_price'] = trade_details['entry_price']
                            data.at[j, 'stop_loss'] = trade_details['stop_loss']
                            data.at[j, 'risk'] = trade_details['risk']
                            data.at[j, 'take_profit_target'] = trade_details['take_profit_target']
                            break  # Exit loop once trade is opened

            ### TRADE TYPE B ###
            elif data['RSI_21'].iloc[i-1] <= oversold_level:
                ranged = False
                for j in range(i, len(data)):
                    if 49.50 <= data['RSI_21'].iloc[j] <= 50.49:
                        ranged = True
                    elif data['RSI_21'].iloc[j] >= overbought_level and ranged:
                        # Trade B: RSI was <= oversold level, ranged, and then crossed overbought level
                        for k in range(j+1, j+4):
                            if data['High'].iloc[k] >= data['High'].iloc[j] + spread:
                                # Check if the condition to place the trade based on risk is also satisfied
                                if (data['High'].iloc[j] + spread - (data['Low'].iloc[j] - spread)) <= max_risk:
                                    trade_details = {
                                        'active': True,
                                        'trade_type': 'B',
                                        'status': 'Buy Trade B Opened',
                                        'entry_price': data['High'].iloc[j] + spread,
                                        'stop_loss': data['Low'].iloc[j] - spread,
                                        'risk': data['High'].iloc[j] + spread - (data['Low'].iloc[j] - spread),
                                        'take_profit_target': data['High'].iloc[j] + spread + (last_daily_atr / daily_atr_divider),
                                        'trade_open_idx': k,
                                        'stop_loss_moved_to_be': False
                                    }
                                    active_trades.append(trade_details)

                                    # Update DataFrame with details of the opened trade at index k
                                    data.at[k, 'trade_open'] = True
                                    data.at[k, 'trade_type'] = 'B'
                                    data.at[k, 'status'] = 'Buy Trade B Opened'
                                    data.at[k, 'entry_price'] = trade_details['entry_price']
                                    data.at[k, 'stop_loss'] = trade_details['stop_loss']
                                    data.at[k, 'risk'] = trade_details['risk']
                                    data.at[k, 'take_profit_target'] = trade_details['take_profit_target']
                                    break  # Exit loop once trade is opened
                        break  # Exit outer loop after handling the condition
                    else:
                        break  # Break if RSI level is not ranged or overbought

### MANAGING OPEN A TRADES ###

    for trade in active_trades:
        if trade['active'] and trade['trade_type'] == 'A':
            if i > trade['trade_open_idx']:

                # Check for loss condition
                if data['Low'].iloc[i] <= trade['stop_loss']:
                    trade['active'] = False
                    trade['stop_loss_moved_to_be'] = False
                    trade['exit_time'] = data['Time'].iloc[i]
                    exit_price = trade['stop_loss']
                    trade['exit_price'] = exit_price
                    pnl = exit_price - trade['entry_price']
                    cumulative_pnl += pnl
                    if pnl < 0:
                        outcome = 'Loss'
                        daily_losses += 1
                        total_losses += 1
                    elif pnl == 0:
                        outcome = 'B/E'

                    # Update the DataFrame with trade details
                    data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                    data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                    data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                    data.at[trade['trade_open_idx'], 'exit_time'] = trade['exit_time']
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                    data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                    data.at[trade['trade_open_idx'], 'outcome'] = outcome
                    data.at[trade['trade_open_idx'], 'PnL'] = pnl
                    data.at[trade['trade_open_idx'], 'daily_losses'] = daily_losses
                    data.at[trade['trade_open_idx'], 'total_wins'] = total_wins
                    data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                # Move stop loss to break-even if conditions are met
                if not trade['stop_loss_moved_to_be'] and data['High'].iloc[i] >= trade['entry_price'] + stop_loss_to_BE:
                    trade['stop_loss'] = trade['entry_price']
                    trade['stop_loss_moved_to_be'] = True
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['entry_price']

                # Check for take profit condition based up a percentage of the Daily ATR
                # if data['High'].iloc[i] >= trade['take_profit_target']:
                #     trade['active'] = False
                #     trade['stop_loss_moved_to_be'] = False
                #     exit_price = trade['take_profit_target']
                #     trade['exit_price'] = exit_price
                #     trade['exit_time'] = data['Time'].iloc[i]
                #     pnl = exit_price - trade['entry_price']
                #     cumulative_pnl += pnl
                #     outcome = 'Win'
                #     total_wins += 1

                #     data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                #     data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                #     data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                #     data.at[trade['trade_open_idx'], 'exit_time'] = trade['exit_time']
                #     data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                #     data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                #     data.at[trade['trade_open_idx'], 'outcome'] = outcome
                #     data.at[trade['trade_open_idx'], 'PnL'] = pnl
                #     data.at[trade['trade_open_idx'], 'daily_losses'] = daily_losses
                #     data.at[trade['trade_open_idx'], 'total_wins'] = total_wins
                #     data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                # Check for take profit condition based up a the RSI crossing back over to Oversold
                if data['RSI_21'].iloc[i] <= oversold_level:
                    take_profit_met = False
                    for j in range(i + 1, len(data)):
                        if data['RSI_21'].iloc[j] >= overbought_level:
                            break

                        if data['Low'].iloc[j] <= data['Low'].iloc[i] - spread:
                            potential_pnl = (data['Low'].iloc[i] - spread) - trade['entry_price']
                            if potential_pnl >= 1:
                                take_profit_met = True
                                trade['exit_price'] = data['Low'].iloc[i] - spread
                                break

                    if take_profit_met:
                        trade['active'] = False
                        trade['stop_loss_moved_to_be'] = False
                        pnl = trade['exit_price'] - trade['entry_price']
                        cumulative_pnl += pnl
                        trade['outcome'] = 'Win'
                        trade['exit_time'] = data['Time'].iloc[i]
                        total_wins += 1

                        # Update data DataFrame with trade exit details
                        data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                        data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                        data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                        data.at[trade['trade_open_idx'], 'exit_time'] = trade['exit_time']
                        data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                        data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                        data.at[trade['trade_open_idx'], 'outcome'] = trade['outcome']
                        data.at[trade['trade_open_idx'], 'PnL'] = pnl
                        data.at[trade['trade_open_idx'], 'daily_losses'] = daily_losses
                        data.at[trade['trade_open_idx'], 'total_wins'] = total_wins
                        data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                # Close all open trades at the end of the trading day, for win, breakeven or loss.
                if data['Time'].iloc[i] == eod:
                    # Set initial exit price to closing price
                    exit_price = data['Close'].iloc[i]

                    # Check if the stop loss should trigger
                    if data['Low'].iloc[i] <= trade['stop_loss']:
                        exit_price = trade['stop_loss']

                    # Update exit price and calculate PnL
                    trade['exit_price'] = exit_price
                    pnl = exit_price - trade['entry_price']
                    cumulative_pnl += pnl
                    trade['exit_time'] = data['Time'].iloc[i]
                    trade['active'] = False
                    trade['stop_loss_moved_to_be'] = False

                    # Determine the outcome based on pnl
                    if pnl > 0:
                        outcome = 'Win'
                        total_wins += 1
                    elif pnl < 0:
                        outcome = 'Loss'
                        daily_losses += 1
                        total_losses += 1
                    else:
                        outcome = 'B/E'

                    # Update the DataFrame with trade details
                    data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                    data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                    data.at[trade['trade_open_idx'], 'exit_time'] = trade['exit_time']
                    data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                    data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                    data.at[trade['trade_open_idx'], 'outcome'] = outcome
                    data.at[trade['trade_open_idx'], 'PnL'] = pnl
                    data.at[trade['trade_open_idx'], 'daily_losses'] = daily_losses
                    data.at[trade['trade_open_idx'], 'total_wins'] = total_wins
                    data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                    # Reset active trades list after closing all trades
                    active_trades = []
                    continue

### MANAGING OPEN B TRADES ###

        if trade['active'] and trade['trade_type'] == 'B':
            if i > trade['trade_open_idx']:

                # Check for loss condition
                if data['Low'].iloc[i] <= trade['stop_loss']:
                    trade['active'] = False
                    trade['stop_loss_moved_to_be'] = False
                    exit_price = trade['stop_loss']
                    trade['exit_price'] = exit_price
                    trade['exit_time'] = data['Time'].iloc[i]
                    pnl = exit_price - trade['entry_price']
                    cumulative_pnl += pnl
                    if pnl < 0:
                        outcome = 'Loss'
                        daily_losses += 1
                        total_losses += 1
                    elif pnl == 0:
                        outcome = 'B/E'

                    # Update the DataFrame with trade details
                    data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                    data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                    data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                    data.at[trade['trade_open_idx'], 'exit_time'] = trade['exit_time']
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                    data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                    data.at[trade['trade_open_idx'], 'outcome'] = outcome
                    data.at[trade['trade_open_idx'], 'PnL'] = pnl
                    data.at[trade['trade_open_idx'], 'daily_losses'] = daily_losses
                    data.at[trade['trade_open_idx'], 'total_wins'] = total_wins
                    data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                # Move stop loss to break-even if conditions are met
                if not trade['stop_loss_moved_to_be'] and data['High'].iloc[i] >= trade['entry_price'] + stop_loss_to_BE:
                    trade['stop_loss'] = trade['entry_price']
                    trade['stop_loss_moved_to_be'] = True
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['entry_price']

                # Check for take profit condition based up a percentage of the Daily ATR
                # if data['High'].iloc[i] >= trade['take_profit_target']:
                #     trade['active'] = False
                #     trade['stop_loss_moved_to_be'] = False
                #     exit_price = trade['take_profit_target']
                #     trade['exit_price'] = exit_price
                #     trade['exit_time'] = data['Time'].iloc[i]
                #     pnl = exit_price - trade['entry_price']
                #     cumulative_pnl += pnl
                #     outcome = 'Win'
                #     total_wins += 1

                #     data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                #     data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                #     data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                #     data.at[trade['trade_open_idx'], 'exit_time'] = trade['exit_time']
                #     data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                #     data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                #     data.at[trade['trade_open_idx'], 'outcome'] = outcome
                #     data.at[trade['trade_open_idx'], 'PnL'] = pnl
                #     data.at[trade['trade_open_idx'], 'daily_losses'] = daily_losses
                #     data.at[trade['trade_open_idx'], 'total_wins'] = total_wins
                #     data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                # Check for take profit condition based up a the RSI crossing back over to Oversold
                if data['RSI_21'].iloc[i] <= oversold_level:
                    take_profit_met = False
                    for j in range(i + 1, len(data)):
                        if data['RSI_21'].iloc[j] >= overbought_level:
                            break

                        if data['Low'].iloc[j] <= data['Low'].iloc[i] - spread:
                            potential_pnl = (data['Low'].iloc[i] - spread) - trade['entry_price']
                            if potential_pnl >= 1:
                                take_profit_met = True
                                trade['exit_price'] = data['Low'].iloc[i] - spread
                                break

                    if take_profit_met:
                        trade['active'] = False
                        trade['stop_loss_moved_to_be'] = False
                        pnl = trade['exit_price'] - trade['entry_price']
                        cumulative_pnl += pnl
                        trade['outcome'] = 'Win'
                        trade['exit_time'] = data['Time'].iloc[i]
                        total_wins += 1

                        # Update data DataFrame with trade exit details
                        data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                        data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                        data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                        data.at[trade['trade_open_idx'], 'exit_time'] = trade['exit_time']
                        data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                        data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                        data.at[trade['trade_open_idx'], 'outcome'] = trade['outcome']
                        data.at[trade['trade_open_idx'], 'PnL'] = pnl
                        data.at[trade['trade_open_idx'], 'daily_losses'] = daily_losses
                        data.at[trade['trade_open_idx'], 'total_wins'] = total_wins
                        data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                # Close all open trades at the end of the trading day, for win, breakeven or loss.
                if data['Time'].iloc[i] == eod:
                    exit_price = data['Close'].iloc[i]

                    # Check if the stop loss should trigger
                    if data['Low'].iloc[i] <= trade['stop_loss']:
                        exit_price = trade['stop_loss']

                    trade['exit_price'] = exit_price
                    pnl = exit_price - trade['entry_price']
                    cumulative_pnl += pnl
                    trade['exit_time'] = data['Time'].iloc[i]
                    trade['active'] = False
                    trade['stop_loss_moved_to_be'] = False

                    # Determine the outcome based on pnl
                    if pnl > 0:
                        outcome = 'Win'
                        total_wins += 1
                    elif pnl < 0:
                        outcome = 'Loss'
                        daily_losses += 1
                        total_losses += 1
                    else:
                        outcome = 'B/E'

                    # Update the DataFrame with trade details
                    data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                    data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                    data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                    data.at[trade['trade_open_idx'], 'exit_time'] = trade['exit_time']
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                    data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                    data.at[trade['trade_open_idx'], 'outcome'] = outcome
                    data.at[trade['trade_open_idx'], 'PnL'] = pnl
                    data.at[trade['trade_open_idx'], 'daily_losses'] = daily_losses
                    data.at[trade['trade_open_idx'], 'total_wins'] = total_wins
                    data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                    # Reset active trades list after closing all trades
                    active_trades = []
                    continue

### CALCULATING PNL AND STRATEGY RESULTS ###

# Calculate cumulative PnL across the DataFrame
data['cumulative_PnL'] = data['PnL'].cumsum()

# Create a figure for the cumulative PnL chart
fig_pnl = go.Figure()

# Add the PnL line
fig_pnl.add_trace(go.Scatter(x=data['Datetime'], y=data['cumulative_PnL'], mode='lines', name='Cumulative PnL', line=dict(color='green')))

# Add background shading for continuous periods with RSI_21_Daily >= overbought_level or <= oversold_level
overbought_periods = data['RSI_21_Daily'] >= overbought_level
oversold_periods = data['RSI_21_Daily'] <= oversold_level

# Initialize variables to track the start of shading periods
start_overbought = None
start_oversold = None

for i in range(len(data)):
    # Check for start of overbought period
    if overbought_periods.iloc[i] and start_overbought is None:
        start_overbought = data['Datetime'].iloc[i]
    # Check for end of overbought period
    elif not overbought_periods.iloc[i] and start_overbought is not None:
        fig_pnl.add_shape(type='rect',
                          xref='x', yref='paper',
                          x0=start_overbought, y0=0,
                          x1=data['Datetime'].iloc[i], y1=1,
                          fillcolor='green', opacity=0.2, line_width=0)
        start_overbought = None

    # Check for start of oversold period
    if oversold_periods.iloc[i] and start_oversold is None:
        start_oversold = data['Datetime'].iloc[i]
    # Check for end of oversold period
    elif not oversold_periods.iloc[i] and start_oversold is not None:
        fig_pnl.add_shape(type='rect',
                          xref='x', yref='paper',
                          x0=start_oversold, y0=0,
                          x1=data['Datetime'].iloc[i], y1=1,
                          fillcolor='red', opacity=0.2, line_width=0)
        start_oversold = None

# Handle the case where a period extends to the end of the data
if start_overbought is not None:
    fig_pnl.add_shape(type='rect',
                      xref='x', yref='paper',
                      x0=start_overbought, y0=0,
                      x1=data['Datetime'].iloc[-1], y1=1,
                      fillcolor='green', opacity=0.2, line_width=0)

if start_oversold is not None:
    fig_pnl.add_shape(type='rect',
                      xref='x', yref='paper',
                      x0=start_oversold, y0=0,
                      x1=data['Datetime'].iloc[-1], y1=1,
                      fillcolor='red', opacity=0.2, line_width=0)

# Set the layout for the PnL chart
fig_pnl.update_layout(title='Cumulative Profit and Loss over Time',
                      xaxis_title='Datetime',
                      yaxis_title='Cumulative PnL',
                      showlegend=True)

# Display the PnL chart
fig_pnl.show()


print("Total Wins:", total_wins)
print("Total Losses:", total_losses)
if total_wins + total_losses > 0:
    win_percentage = (total_wins / (total_wins + total_losses)) * 100
    print("Win/Loss Percentage: {:.2f}%".format(win_percentage))
else:
    print("No completed trades to calculate win/loss percentage.")
print("Biggest Win: {:.2f} Pts".format(data['PnL'].max()))
print("Biggest Loss: {:.2f} Pts".format(data['PnL'].min()))
print("Cumulative PnL: {:.2f} Pts".format(cumulative_pnl))
print("Starting balance: £", starting_balance)
print("Cash returns @ £{} per pt: £{:.2f}".format(per_pt, cumulative_pnl * per_pt))

cash_returns = cumulative_pnl * per_pt
final_balance = starting_balance + cash_returns
percentage_returns = ((final_balance - starting_balance) / starting_balance) * 100

print("Percentage returns: {:.2f}%".format(percentage_returns))

# Drop unnecessary columns
data.drop(['Change', 'Gain', 'Loss', 'AvgGain', 'AvgLoss', 'RS', 'range', 'ATR', 'trade_open'], axis=1, inplace=True)

### EXPORT STRATEGY RESULTS AND DATAFRAME TO EXCEL ###
data.to_excel("RSI_Results_Buy.xlsx")
