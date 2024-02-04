

# .\my_venv\Scripts\activate

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
data = pd.read_csv("DJIA_Data_CSV.csv")

# Ensure the 'Date' and 'Time' columns are combined into a datetime column
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

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
                        xaxis_rangeslider_visible=False)  # Hide the rangeslider for simplicity

# Display the price chart
fig_price.show()

# Read data from CSV file
data = pd.read_csv("DJIA_Data_CSV.csv")

# Combine 'Date' and 'Time' columns to create a datetime column
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

# Assuming 'rsiw' is defined as the input window size
rsiw = 21
overbought_level = 50.5  # Adjust as needed
oversold_level = 49.49  # Adjust as needed

# Compute change, gain, and loss
data['Change'] = data['Close'].diff()
data['Gain'] = data['Change'].apply(lambda x: x if x > 0 else 0)
data['Loss'] = data['Change'].apply(lambda x: abs(x) if x < 0 else 0)

# Calculate initial AvgGain and AvgLoss for the 22nd row
data.loc[21, 'AvgGain'] = data['Gain'].iloc[:22].mean()
data.loc[21, 'AvgLoss'] = data['Loss'].iloc[:22].mean()

# Calculate AvgGain and AvgLoss for subsequent rows
for i in range(22, len(data)):
    data.loc[i, 'AvgGain'] = ((data.loc[i-1, 'AvgGain'] * 20) + data.loc[i, 'Gain']) / 21
    data.loc[i, 'AvgLoss'] = ((data.loc[i-1, 'AvgLoss'] * 20) + data.loc[i, 'Loss']) / 21

# Compute relative strength
data['RS'] = data['AvgGain'] / data['AvgLoss']

# Compute RSI values
data['RSI_21'] = 100 - (100 / (1 + data['RS']))

# Drop NaN values introduced by the rolling mean
data = data.dropna()

# Plot RSI values
fig = go.Figure()

fig.add_trace(go.Scatter(x=data['DateTime'], y=data['RSI_21'], mode='lines', name='RSI_21'))
fig.add_trace(go.Scatter(x=data['DateTime'], y=[overbought_level] * len(data['RSI_21']), mode='lines', name='Overbought', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=data['DateTime'], y=[oversold_level] * len(data['RSI_21']), mode='lines', name='Oversold', line=dict(dash='dash')))

fig.update_layout(title='RSI Chart', xaxis_title='Date and Time', yaxis_title='RSI')
fig.show()


atr_period = 5  # defining the ATR period to 5

# calculating the range of each candle
data['range'] = data['High'] - data['Low']

# calculating the average value of ranges
data['atr_5'] = data['range'].rolling(atr_period).mean()
\
# Display the DataFrame with Time and ATR values
# print(data[['Time', 'atr_5']])

# Plot ATR values
fig = go.Figure()

fig.add_trace(go.Scatter(x=data['DateTime'], y=data['atr_5'], mode='lines', name='ATR'))
fig.update_layout(title='Average True Range (ATR) Chart', xaxis_title='Date and Time', yaxis_title='ATR')
fig.show()

trade_log = []
trade_open = False
trade_type = None  # 'buy' or 'sell'
buy_entry_price = None
buy_stop_loss = None
buy_exit_price = None
Trade_outcome = None

# Iterate through DataFrame to find trade signals
for i in range(1, len(data)):
    if not trade_open:
        # Check for buy signal
        if data['RSI_21'].iloc[i-1] <= oversold_level and data['RSI_21'].iloc[i] >= overbought_level:
            # Confirm buy condition A
            if sum(data['High'].iloc[i+1:i+4] >= data['High'].iloc[i] + 2):
                trade_open = True
                trade_type = 'Buy'
                buy_entry_price = data['High'].iloc[i] + 2
                buy_stop_loss = data['Low'].iloc[i] - 2
                trade_log.append(('Buy Trade Open', buy_entry_price, buy_stop_loss, None, None, data['Time'].iloc[i]))
                continue

    else:
        # Trade management and exit conditions
        if trade_type == 'Buy':
            # Check for buy signal loss
            if data['Low'].iloc[i] <= buy_stop_loss:
                trade_open = False
                buy_exit_price = buy_stop_loss
                Trade_outcome = 'Loss'
                trade_log.append(('Buy Trade Closed', buy_entry_price, buy_stop_loss, buy_exit_price, Trade_outcome, data['Time'].iloc[i]))
                continue

        # Check for buy signal take profit
        if data['RSI_21'].iloc[i] <= oversold_level and (data['Low'].iloc[i+1:] <= data['Low'].iloc[i] - 2).any():
            potential_buy_exit_price = data['Low'].iloc[i] - 2  # Potential exit price
            pnl = potential_buy_exit_price - buy_entry_price  # Calculate PnL

            if pnl >= 1:  # Check if PnL is greater than or equal to 1
                trade_open = False
                buy_exit_price = potential_buy_exit_price  # Use the potential exit price as the actual exit price
                Trade_outcome = 'Win'
                trade_log.append(('Buy Trade Closed', buy_entry_price, None, buy_exit_price, Trade_outcome, data['Time'].iloc[i]))
                continue

        # Check if the current time is 20:59:00 and a trade is open
        if data['Time'].iloc[i] == "15:59:00" and trade_open:
            trade_open = False
            exit_price = data['Close'].iloc[i]  # Ensure exit_price is assigned from the 'Close' column

            if trade_type == 'Buy':
                if buy_entry_price is not None and exit_price is not None:  # Ensure both prices are not None
                    pnl = exit_price - buy_entry_price  # Calculate PnL for buy trade
                    Trade_outcome = 'Win' if pnl > 0 else 'Loss'
                    trade_log.append(('Buy Trade Closed at EOD', buy_entry_price, buy_stop_loss, exit_price, Trade_outcome, data['Time'].iloc[i]))


# Loop to populate new columns in 'data' DataFrame from 'trade_log'
for trade in trade_log:
    # Extract information from each trade log entry
    buy_trade_open, buy_entry_price, buy_stop_loss, buy_exit_price, trade_outcome, datetime = trade
    
    # Find the corresponding row in 'data' DataFrame
    row_index = data.index[data['Time'] == datetime].tolist()
    if row_index:
        # Populate buy trade details
        data.at[row_index[0], 'Buy Trade Open'] = buy_trade_open
        data.at[row_index[0], 'Buy Entry Price'] = buy_entry_price
        data.at[row_index[0], 'Buy Stop Loss'] = buy_stop_loss
        data.at[row_index[0], 'Buy Exit Price'] = buy_exit_price      
        data.at[row_index[0], 'Trade Outcome'] = trade_outcome
        
        # Calculate PnL for buy trades
        if buy_entry_price is not None and buy_exit_price is not None:
            data.at[row_index[0], 'PnL'] = buy_exit_price - buy_entry_price
        
# Calculate Cumulative PnL
data['PnL'] = data['PnL'].fillna(0)  # Replace None with 0 for calculation
data['Cumulative PnL'] = data['PnL'].cumsum()  # Calculate cumulative sum of PnL

# Plot cumulative PnL over time
import plotly.express as px
fig = px.line(data, x='Time', y='Cumulative PnL', title='Cumulative PnL Over Time')
fig.show()

# Display part of the DataFrame to check the results
print(data.tail())  # Display the last few rows for checking

# Columns to exclude from the export
columns_to_exclude = ['Change', 'Gain', 'Loss', 'AvgGain', 'AvgLoss', 'RS', 'range', 'atr_5']

# Create a copy of the DataFrame without the specified columns
data_to_export = data.drop(columns=columns_to_exclude, errors='ignore')

# Export the modified DataFrame to Excel, excluding the specified columns
data_to_export.to_excel("Buy_Backtest_Results.xlsx", index=False)