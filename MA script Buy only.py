

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

# Read CSV file into a DataFrame
data = pd.read_csv("DJIA_DAILY_DATA.csv")

# Ensure the 'Date' column is in the appropriate datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Calculate the simple moving averages for 50 and 200 periods
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()


### DISPLAYING THE PRICE BAR CHART AND MA LINES ###

# Create the price bar chart
fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlestick')])

# Add the 50-day moving average line
fig.add_trace(go.Scatter(x=data['Date'], y=data['50_MA'], name='50_MA', line=dict(color='blue', width=1.5)))

# Add the 200-day moving average line
fig.add_trace(go.Scatter(x=data['Date'], y=data['200_MA'], name='200_MA', line=dict(color='green', width=1.5)))

# Set the layout for the chart
fig.update_layout(title='Moving Averages with Daily DJIA Data',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False) 

# Display the chart
fig.show()

### CALCULATING AND DISPLAYING THE ATR ###

atr_period = 5 

# calculating the range
data['Range'] = data['High'] - data['Low']

# calculating the average value of ranges
data['ATR'] = data['Range'].rolling(atr_period).mean()

# Plot ATR values
fig = go.Figure()

### CALCULATING THE BACKTESTING LOGIC ###

# Creating columns
data['status'] =  None
data['trade_type'] =  None
data['entry_price'] = np.nan 
data['stop_loss'] = np.nan
data['exit_price'] = np.nan
data['exit_date'] = None
data['take_profit_target'] = np.nan
data['risk'] = np.nan
data['outcome'] = None
data['PnL'] = 0.0
data['Wins'] = None
data['total_losses'] = None

# Initialize variables for trade tracking
trade_open = False
total_losses = 0  # Counter for total losses
total_wins = 0  # Counter for total wins
exit_date = None
cumulative_pnl = 0  # Cumulative PnL
stop_loss_moved_to_be = False

# Trade variables
starting_balance = 100000
per_pt = 20
max_risk = 50000
spread = 10
atr_take_profit_multiplier = 3
atr_stop_loss_multiplier = 1.5
stop_loss_to_BE = 30

# Creating a List to hold active trades
active_trades = []


for i in range(len(data)):
    # Check if all active trades have their stop loss at BE, allowing for a new trade to be opened
    can_open_new_trade = True
    for trade in active_trades:
        if trade['active'] and not trade['stop_loss_moved_to_be']:
            can_open_new_trade = False
            break

        ### OPENING TRADES LOGIC ###

    if can_open_new_trade:
        #### TRADE TYPE A ###
        # Trade A: Daily bar closes above 200 Period MA
        if data['Close'].iloc[i] > data['Open'].iloc[i] and data['Close'].iloc[i] >= data['200_MA'].iloc[i] + 1:
            if data['Close'].iloc[i-1] <= data['200_MA'].iloc[i-1] - 1 or data['Low'].iloc[i] <= data['200_MA'].iloc[i] - 1:
                for j in range(i+1, i+4):
                    if data['High'].iloc[j] >= data['High'].iloc[i] + spread:
                        # Check if the condition to place the trade based on risk is also satisfied
                        if (data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread)) <= max_risk:
                            # Execute trade
                            trade_details = {
                                'active': True,
                                'trade_type': 'A',
                                'status': 'Buy Trade A Opened',
                                'stop_loss_moved_to_be': False,
                                'entry_price': data['High'].iloc[i] + spread,
                                'stop_loss': data['Low'].iloc[i] - spread,
                                'risk': data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread),
                                'take_profit_target': data['High'].iloc[i] + (data['ATR'].iloc[i] * atr_take_profit_multiplier),
                                'trade_open_idx': j
                            }
                            active_trades.append(trade_details)

                            # Update data DataFrame with trade details at index j
                            data.at[j, 'trade_type'] = 'A'
                            data.at[j, 'status'] = 'Buy Trade A Opened'
                            data.at[j, 'entry_price'] = trade_details['entry_price']
                            data.at[j, 'stop_loss'] = trade_details['stop_loss']
                            data.at[j, 'risk'] = trade_details['risk']
                            data.at[j, 'take_profit_target'] = trade_details['take_profit_target']
                            break  # Exit loop once trade is opened

        ### TRADE TYPE B ###
        # Trade B: Daily bar closes above 50 Period MA while also above the 200 Period MA
        if data['Close'].iloc[i] >= data['200_MA'].iloc[i] + 1 and \
           data['Close'].iloc[i] > data['Open'].iloc[i] and \
           data['Close'].iloc[i] >= data['50_MA'].iloc[i] + 1:
            if data['Close'].iloc[i-1] <= data['50_MA'].iloc[i-1] - 1:
                for j in range(i+1, i+4):
                    if data['High'].iloc[j] >= data['High'].iloc[i] + spread:
                        # Check if the condition to place the trade based on risk is also satisfied
                        if (data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread)) <= max_risk:
                            # Execute Trade
                            trade_details = {
                                'active': True,
                                'trade_type': 'B',
                                'status': 'Buy Trade B Opened',
                                'stop_loss_moved_to_be': False,
                                'entry_price': data['High'].iloc[i] + spread,
                                'stop_loss': data['Low'].iloc[i] - spread,
                                'risk': data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread),
                                'take_profit_target': data['High'].iloc[i] + (data['ATR'].iloc[i] * atr_take_profit_multiplier),
                                'trade_open_idx': j
                            }
                            active_trades.append(trade_details)

                            # Update data DataFrame with trade details at index j
                            data.at[j, 'trade_type'] = 'B'
                            data.at[j, 'status'] = 'Buy Trade B Opened'
                            data.at[j, 'entry_price'] = trade_details['entry_price']
                            data.at[j, 'stop_loss'] = trade_details['stop_loss']
                            data.at[j, 'risk'] = trade_details['risk']
                            data.at[j, 'take_profit_target'] = trade_details['take_profit_target']
                            break  # Exit loop once trade is opened


### MANAGING OPEN A TRADES ###

    for trade in active_trades:
        take_profit_met = False
        if trade['active'] and trade['trade_type'] == 'A':
            if i > trade['trade_open_idx']:

                # Check for loss condition
                if data['Low'].iloc[i] <= trade['stop_loss']:
                    trade['active'] = False
                    trade['stop_loss_moved_to_be'] = False
                    trade['exit_date'] = data['Date'].iloc[i]
                    exit_price = trade['stop_loss']
                    trade['exit_price'] = exit_price
                    pnl = exit_price - trade['entry_price']
                    cumulative_pnl += pnl
                    if pnl < 0:
                        outcome = 'Loss'
                        total_losses += 1
                    elif pnl == 0:
                        outcome = 'B/E'

                    # Update the DataFrame with trade details
                    data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                    data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                    data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                    data.at[trade['trade_open_idx'], 'exit_date'] = trade['exit_date']
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                    data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                    data.at[trade['trade_open_idx'], 'outcome'] = outcome
                    data.at[trade['trade_open_idx'], 'PnL'] = pnl
                    data.at[trade['trade_open_idx'], 'Wins'] = total_wins
                    data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                # Move stop loss to break-even if conditions are met
                if not trade['stop_loss_moved_to_be'] and data['High'].iloc[i] >= trade['entry_price'] + data['ATR'].iloc[i] * atr_stop_loss_multiplier: #stop_loss_to_BE
                    trade['stop_loss'] = trade['entry_price']
                    trade['stop_loss_moved_to_be'] = True
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['entry_price']

                # Check for take profit condition based up a multiple of the ATR
                # if data['High'].iloc[i] >= trade['take_profit_target']:
                #     trade['active'] = False
                #     trade['stop_loss_moved_to_be'] = False
                #     exit_price = trade['take_profit_target']
                #     trade['exit_price'] = exit_price
                #     trade['exit_date'] = data['Date'].iloc[i]
                #     pnl = exit_price - trade['entry_price']
                #     cumulative_pnl += pnl
                #     outcome = 'Win'
                #     total_wins += 1

                #     data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                #     data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                #     data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                #     data.at[trade['trade_open_idx'], 'exit_date'] = trade['exit_date']
                #     data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                #     data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                #     data.at[trade['trade_open_idx'], 'outcome'] = outcome
                #     data.at[trade['trade_open_idx'], 'PnL'] = pnl
                #     data.at[trade['trade_open_idx'], 'Wins'] = total_wins
                #     data.at[trade['trade_open_idx'], 'total_losses'] = total_losses


                # Check for take profit condition based up a the daily bar closing back below the 200 period MA
                if data['Close'].iloc[i] < data['Open'].iloc[i] and data['Close'].iloc[i] <= data['200_MA'].iloc[i] - 1:
                    if data['High'].iloc[i] >= data['200_MA'].iloc[i] + 1 or data['High'].iloc[i-1] >= data['200_MA'].iloc[i-1] + 1:
                        
                        for j in range(i + 1, len(data)):
                            if data['High'].iloc[j] >= data['200_MA'].iloc[j] + 1:
                                break

                            if data['Low'].iloc[j] <= data['Low'].iloc[i] - spread:
                                # Calculate potential PnL for this exit price
                                potential_pnl = (data['Low'].iloc[i] - spread) - trade['entry_price']
                                
                                if potential_pnl >= 1:
                                    take_profit_met = True
                                    trade['exit_price'] = data['Low'].iloc[i] - spread
                                    break

                        if take_profit_met:
                            trade['active'] = False
                            trade['stop_loss_moved_to_be'] = False
                            trade['exit_date'] = data['Date'].iloc[i]
                            trade['exit_price'] = data['Low'].iloc[i] - spread
                            trade['outcome'] = 'Win'
                            pnl = trade['exit_price'] - trade['entry_price']
                            trade['pnl'] = pnl
                            cumulative_pnl += pnl
                            total_wins += 1

                            # Update data DataFrame with trade exit details
                            data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                            data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                            data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                            data.at[trade['trade_open_idx'], 'exit_date'] = trade['exit_date']
                            data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                            data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                            data.at[trade['trade_open_idx'], 'outcome'] = trade['outcome']
                            data.at[trade['trade_open_idx'], 'PnL'] = trade['pnl']
                            data.at[trade['trade_open_idx'], 'Wins'] = total_wins
                            data.at[trade['trade_open_idx'], 'total_losses'] = total_losses
                            continue  # Exit the loop since the trade is closed

### MANAGING OPEN B TRADES ###

        if trade['active'] and trade['trade_type'] == 'B':
            if i > trade['trade_open_idx']:                         

                # Check for loss condition
                if data['Low'].iloc[i] <= trade['stop_loss']:
                    trade['active'] = False
                    trade['stop_loss_moved_to_be'] = False
                    trade['exit_date'] = data['Date'].iloc[i]
                    exit_price = trade['stop_loss']
                    trade['exit_price'] = exit_price
                    pnl = exit_price - trade['entry_price']
                    cumulative_pnl += pnl
                    if pnl < 0:
                        outcome = 'Loss'
                        total_losses += 1
                    elif pnl == 0:
                        outcome = 'B/E'

                    # Update the DataFrame with trade details
                    data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                    data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                    data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                    data.at[trade['trade_open_idx'], 'exit_date'] = trade['exit_date']
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                    data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                    data.at[trade['trade_open_idx'], 'outcome'] = outcome
                    data.at[trade['trade_open_idx'], 'PnL'] = pnl
                    data.at[trade['trade_open_idx'], 'Wins'] = total_wins
                    data.at[trade['trade_open_idx'], 'total_losses'] = total_losses

                # Move stop loss to break-even if conditions are met
                if not trade['stop_loss_moved_to_be'] and data['High'].iloc[i] >= trade['entry_price'] + data['ATR'].iloc[i] * atr_stop_loss_multiplier:  #stop_loss_to_BE
                    trade['stop_loss'] = trade['entry_price']
                    trade['stop_loss_moved_to_be'] = True
                    data.at[trade['trade_open_idx'], 'stop_loss'] = trade['entry_price']                            
        

                # Check for take profit condition based up a multiple of the ATR
                # if data['High'].iloc[i] >= trade['take_profit_target']:
                #     trade['active'] = False
                #     trade['stop_loss_moved_to_be'] = False
                #     exit_price = trade['take_profit_target']
                #     trade['exit_price'] = exit_price
                #     trade['exit_date'] = data['Date'].iloc[i]
                #     pnl = exit_price - trade['entry_price']
                #     cumulative_pnl += pnl
                #     outcome = 'Win'
                #     total_wins += 1

                #     data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                #     data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                #     data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                #     data.at[trade['trade_open_idx'], 'exit_date'] = trade['exit_date']
                #     data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                #     data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                #     data.at[trade['trade_open_idx'], 'outcome'] = outcome
                #     data.at[trade['trade_open_idx'], 'PnL'] = pnl
                #     data.at[trade['trade_open_idx'], 'Wins'] = total_wins
                #     data.at[trade['trade_open_idx'], 'total_losses'] = total_losses


                # Check for take profit condition based up a the daily bar closing back below the 50 period MA
                if data['Close'].iloc[i] < data['Open'].iloc[i] and data['Close'].iloc[i] <= data['50_MA'].iloc[i] - 1:
                    if data['High'].iloc[i] >= data['50_MA'].iloc[i] + 1 or data['High'].iloc[i-1] >= data['50_MA'].iloc[i-1] + 1:
                        
                        for j in range(i + 1, len(data)):
                            if data['High'].iloc[j] >= data['50_MA'].iloc[j] + 1:
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
                            trade['exit_price'] = data['Low'].iloc[i] - spread
                            trade['exit_date'] = data['Date'].iloc[i]
                            trade['outcome'] = 'Win'
                            pnl = trade['exit_price'] - trade['entry_price']
                            trade['pnl'] = pnl
                            cumulative_pnl += pnl
                            total_wins += 1

                            # Update data DataFrame with trade exit details
                            data.at[trade['trade_open_idx'], 'trade_type'] = trade['trade_type']
                            data.at[trade['trade_open_idx'], 'entry_price'] = trade['entry_price']
                            data.at[trade['trade_open_idx'], 'exit_price'] = trade['exit_price']
                            data.at[trade['trade_open_idx'], 'exit_date'] = trade['exit_date']
                            data.at[trade['trade_open_idx'], 'stop_loss'] = trade['stop_loss']
                            data.at[trade['trade_open_idx'], 'risk'] = trade['risk']
                            data.at[trade['trade_open_idx'], 'outcome'] = trade['outcome']
                            data.at[trade['trade_open_idx'], 'PnL'] = trade['pnl']
                            data.at[trade['trade_open_idx'], 'Wins'] = total_wins
                            data.at[trade['trade_open_idx'], 'total_losses'] = total_losses
                            continue  # Exit the loop since the trade is closed


### CALCULATING PNL AND STRATEGY RESULTS ###

# Calculate cumulative PnL across the DataFrame
data['cumulative_PnL'] = data['PnL'].cumsum()


plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['cumulative_PnL'], label='Cumulative PnL', color='green')
plt.title('Cumulative Profit and Loss over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.grid(True)
plt.legend()
plt.show()

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
print("Starting balance: £",starting_balance)
print("Cash returns @ £{} per pt: £{:.2f}".format(per_pt, cumulative_pnl * per_pt))

cash_returns = cumulative_pnl * per_pt
final_balance = starting_balance + cash_returns
percentage_returns = ((final_balance - starting_balance) / starting_balance) * 100

print("Percentage returns: {:.2f}%".format(percentage_returns))

# Drop unnecessary columns
data.drop(['Range', 'ATR'], axis=1, inplace=True)

### EXPORT STRATEGY RESULTS AND DATAFRAME TO EXCEL ###

# Export the data DataFrame with trade logs to Excel
data.to_excel("MA_DAILY_RESULTS_BUY.xlsx")


