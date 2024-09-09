

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

# Read CSV file into a DataFrame
data = pd.read_csv("DJIA_DAILY_DATA.csv")

print(data)

# Ensure the 'Date' column is in the appropriate datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Calculate the simple moving averages for 50 and 200 periods
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()

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
                  xaxis_rangeslider_visible=False)  # Hide the rangeslider for simplicity

# Display the chart
fig.show()

atr_period = 5  # defining the ATR period to 5

# calculating the range of each candle
data['Range'] = data['High'] - data['Low']

# calculating the average value of ranges
data['ATR'] = data['Range'].rolling(atr_period).mean()

# Plot ATR values
fig = go.Figure()


data['trade_open'] = None
data['entry_price'] = np.nan 
data['stop_loss'] = np.nan
data['exit_price'] = np.nan
data['exit_date'] = None
data['take_profit_target'] = np.nan  # Initialize the column if it doesn't exist
data['risk'] = np.nan
data['outcome'] = None
data['PnL'] = 0.0
data['Losses'] = 0
data['Wins'] = 0


# Initialize variables for trade tracking
trade_open = False
total_losses = 0  # Counter for daily losses
total_wins = 0
exit_date = None
cumulative_pnl = 0  # Cumulative PnL
stop_loss_moved_to_be = False

# Trade variables
max_risk = 20000
spread = 10
atr_take_profit_multiplier = 2
atr_stop_loss_multiplier = 1
stop_loss_to_BE = 30


for i in range(len(data)):
    if not trade_open:
        #TRADE A OPEN
        if data['Close'].iloc[i] > data['Open'].iloc[i] and data['Close'].iloc[i] >= data['200_MA'].iloc[i] + 1:
            # Check if the previous low was below the moving average adjusted by 1
            if data['Low'].iloc[i-1] <= data['200_MA'].iloc[i-1] - 1 or \
                data['Low'].iloc[i] <= data['200_MA'].iloc[i] - 1:
                # Iterate through the next 3 rows to check if the high exceeds High[i] + spread
                for j in range(i+1, i+4):
                    if data['High'].iloc[j] >= data['High'].iloc[i] + spread:
                        if (data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread)) <= max_risk:
                            # Buy signal conditions met, execute trade at index j
                            trade_open = True
                            stop_loss_moved_to_be = False
                            trade_type = 'A'  # Mark the trade as a direct crossover type
                            buy_entry_price = data['High'].iloc[i] + spread  # Entry price adjusted to the high at index j
                            buy_stop_loss = data['Low'].iloc[i] - spread
                            risk = buy_entry_price - buy_stop_loss
                            trade_open_index = j

                            daily_atr_target = data['ATR'].iloc[i] * atr_take_profit_multiplier  # Use ATR from the same index
                            take_profit_target = buy_entry_price + daily_atr_target

                            # Update data DataFrame with trade details at the specific row
                            data.at[data.index[j], 'trade_open'] = 'Buy Trade Open'
                            data.at[data.index[j], 'trade_type'] = trade_type
                            data.at[data.index[j], 'entry_price'] = buy_entry_price
                            data.at[data.index[j], 'stop_loss'] = buy_stop_loss
                            data.at[data.index[j], 'risk'] = risk
                            data.at[data.index[j], 'outcome'] = None
                            data.at[data.index[j], 'take_profit_target'] = take_profit_target
                            break  # Exit loop once trade is opened


        if data['Close'].iloc[i] >= data['200_MA'].iloc[i] + 1 and \
            data['Close'].iloc[i] > data['Open'].iloc[i] and \
            data['Close'].iloc[i] >= data['50_MA'].iloc[i] + 1:
            if data['Close'].iloc[i-1] <= data['50_MA'].iloc[i-1] - 1:
                # Iterate through the next 3 rows to check if the high exceeds High[i] + spread
                for j in range(i+1, i+4):
                    if data['High'].iloc[j] >= data['High'].iloc[i] + spread:
                        if (data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread)) <= max_risk:
                            # Buy signal conditions met, execute trade at index j
                            trade_open = True
                            stop_loss_moved_to_be = False
                            trade_type = 'B'
                            buy_entry_price = data['High'].iloc[i] + spread  # Adjusted to the high at index j
                            buy_stop_loss = data['Low'].iloc[i] - spread
                            risk = buy_entry_price - buy_stop_loss
                            trade_open_index = j

                            daily_atr_target = data['ATR'].iloc[i] * atr_take_profit_multiplier  # Use ATR from the same index
                            take_profit_target = buy_entry_price + daily_atr_target

                            # Update data DataFrame with trade details at the specific row
                            data.at[data.index[j], 'trade_open'] = 'Buy Trade Open'
                            data.at[data.index[j], 'trade_type'] = trade_type
                            data.at[data.index[j], 'entry_price'] = buy_entry_price
                            data.at[data.index[j], 'stop_loss'] = buy_stop_loss
                            data.at[data.index[j], 'risk'] = risk
                            data.at[data.index[j], 'outcome'] = None
                            data.at[data.index[j], 'take_profit_target'] = take_profit_target
                            break  # Exit loop once trade is opened



    if trade_open:
        take_profit_met = False
        pnl = 0

        # MOVE STOP LOSS TO ENTRY (BREAK EVEN)
        if trade_type == 'A':
            if i > trade_open_index:
                if not stop_loss_moved_to_be and data['High'].iloc[i] >= buy_entry_price + data['ATR'].iloc[i] * atr_stop_loss_multiplier:            #stop_loss_to_BE   #data['ATR'].iloc[i] * atr_stop_loss_multiplier
                    buy_stop_loss = buy_entry_price
                    data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                    data.at[data.index[i], 'trade_open'] = 'Stop loss moved to entry'
                    stop_loss_moved_to_be = True
                
                    # Loss conditions for Trade Type A:
                if data['Low'].iloc[i] <= buy_stop_loss:
                    trade_open = False
                    stop_loss_moved_to_be = False
                    trade_type = 'A'
                    trade_outcome = 'Loss'
                    exit_date = data['Date'].iloc[i]
                    pnl = buy_stop_loss - buy_entry_price
                    cumulative_pnl += pnl
                    if pnl < 0:
                        total_losses += 1

                        # Update data DataFrame with trade exit details
                        data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed'
                        data.at[data.index[i], 'trade_type'] = trade_type
                        data.at[data.index[i], 'exit_price'] = buy_stop_loss
                        data.at[data.index[i], 'exit_date'] = exit_date
                        data.at[data.index[i], 'entry_price'] = buy_entry_price
                        data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                        data.at[data.index[i], 'risk'] = risk
                        data.at[data.index[i], 'outcome'] = trade_outcome
                        data.at[data.index[i], 'PnL'] = pnl
                        data.at[data.index[i], 'daily_losses'] = total_losses

                    if pnl == 0:
                        data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed at B/E'
                        data.at[data.index[i], 'trade_type'] = trade_type
                        data.at[data.index[i], 'exit_price'] = buy_stop_loss
                        data.at[data.index[i], 'entry_price'] = buy_entry_price
                        data.at[data.index[i], 'exit_date'] = exit_date
                        data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                        data.at[data.index[i], 'risk'] = risk
                        data.at[data.index[i], 'outcome'] = 'B/E'
                        data.at[data.index[i], 'PnL'] = pnl
                        data.at[data.index[i], 'daily_losses'] = total_losses

                if data['High'].iloc[i] >= take_profit_target:
                    # Take profit conditions met
                    take_profit_met = True
                    trade_open = False
                    stop_loss_moved_to_be = False
                    trade_type = 'A'
                    trade_outcome = 'Win'
                    exit_date = data['Date'].iloc[i]
                    buy_exit_price = take_profit_target 
                    pnl = buy_exit_price - buy_entry_price
                    cumulative_pnl += pnl
                    total_wins += 1

                    # Update data DataFrame with trade exit details
                    data.at[data.index[i], 'Wins'] = total_wins
                    data.at[data.index[i], 'trade_type'] = trade_type
                    data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed via TP'
                    data.at[data.index[i], 'exit_price'] = buy_exit_price
                    data.at[data.index[i], 'exit_date'] = exit_date
                    data.at[data.index[i], 'outcome'] = 'Win'
                    data.at[data.index[i], 'PnL'] = pnl
                    data.at[data.index[i], 'entry_price'] = buy_entry_price
                    data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                    data.at[data.index[i], 'risk'] = risk

                if data['Close'].iloc[i] < data['Open'].iloc[i] and data['Close'].iloc[i] <= data['200_MA'].iloc[i] - 1:
                    if data['High'].iloc[i] >= data['200_MA'].iloc[i] + 1 or data['High'].iloc[i-1] >= data['200_MA'].iloc[i-1] + 1:

                        for j in range(i + 1, len(data)):
                        # Check if any High before the lower Low becomes >= Upper_KC + 1
                            if data['High'].iloc[j] >= data['200_MA'].iloc[j] + 1:
                                break  # Stop the search and restart from the next suitable i

                            # Check if the Low is 3 points or more lower than the Low at i
                            if data['Low'].iloc[j] <= data['Low'].iloc[i] - spread:
                                # Calculate potential PnL for this exit price
                                potential_pnl = (data['Low'].iloc[i] - spread) - buy_entry_price

                                # Check if potential PnL is 1 point or higher
                                if potential_pnl >= 1:
                                    take_profit_met = True
                                    Buy_exit_price = data['Low'].iloc[i] - spread
                                    break 

                        # # If the take profit condition is met and potential PnL is satisfactory, update trade details and PnL
                        if take_profit_met:
                            trade_outcome = 'Win'
                            trade_type = 'A'
                            pnl = Buy_exit_price - buy_entry_price
                            cumulative_pnl += pnl
                            total_wins += 1
                            exit_date = data['Date'].iloc[i]
                            trade_open = False
                            stop_loss_moved_to_be = False

                            # Update data DataFrame with trade exit details
                            data.at[data.index[i], 'Wins'] = total_wins
                            data.at[data.index[i], 'trade_type'] = trade_type
                            data.at[data.index[i], 'exit_date'] = exit_date
                            data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed'
                            data.at[data.index[i], 'exit_price'] = buy_exit_price
                            data.at[data.index[i], 'outcome'] = trade_outcome             
                            data.at[data.index[i], 'PnL'] = pnl
                            data.at[data.index[i], 'entry_price'] = buy_entry_price
                            data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                            data.at[data.index[i], 'risk'] = risk
                            continue

        if trade_type == 'B':
            if i > trade_open_index:
                if not stop_loss_moved_to_be and data['High'].iloc[i] >= buy_entry_price + data['ATR'].iloc[i] * atr_stop_loss_multiplier:            #stop_loss_to_BE   #data['ATR'].iloc[i] * atr_stop_loss_multiplier
                    buy_stop_loss = buy_entry_price
                    data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                    data.at[data.index[i], 'trade_open'] = 'Stop loss moved to entry'
                    stop_loss_moved_to_be = True
                

                    # Loss conditions for Trade Type B:
                if data['Low'].iloc[i] <= buy_stop_loss:
                    trade_open = False
                    stop_loss_moved_to_be = False
                    trade_type = 'B'
                    trade_outcome = 'Loss'
                    exit_date = data['Date'].iloc[i]
                    pnl = buy_stop_loss - buy_entry_price
                    cumulative_pnl += pnl
                    if pnl < 0:
                        total_losses += 1

                        # Update data DataFrame with trade exit details
                        data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed'
                        data.at[data.index[i], 'trade_type'] = trade_type
                        data.at[data.index[i], 'exit_price'] = buy_stop_loss
                        data.at[data.index[i], 'entry_price'] = buy_entry_price
                        data.at[data.index[i], 'exit_date'] = exit_date
                        data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                        data.at[data.index[i], 'risk'] = risk
                        data.at[data.index[i], 'outcome'] = trade_outcome
                        data.at[data.index[i], 'PnL'] = pnl
                        data.at[data.index[i], 'daily_losses'] = total_losses

                    if pnl == 0:
                        data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed at B/E'
                        data.at[data.index[i], 'trade_type'] = trade_type
                        data.at[data.index[i], 'exit_price'] = buy_stop_loss
                        data.at[data.index[i], 'entry_price'] = buy_entry_price
                        data.at[data.index[i], 'exit_date'] = exit_date
                        data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                        data.at[data.index[i], 'risk'] = risk
                        data.at[data.index[i], 'outcome'] = 'B/E'
                        data.at[data.index[i], 'PnL'] = pnl
                        data.at[data.index[i], 'daily_losses'] = total_losses


                if data['High'].iloc[i] >= take_profit_target:
                    # Take profit conditions met
                    take_profit_met = True
                    trade_open = False
                    stop_loss_moved_to_be = False
                    trade_type = 'B'
                    trade_outcome = 'Win'
                    exit_date = data['Date'].iloc[i]
                    buy_exit_price = take_profit_target 
                    pnl = buy_exit_price - buy_entry_price
                    cumulative_pnl += pnl
                    total_wins += 1

                    # Update data DataFrame with trade exit details
                    data.at[data.index[i], 'Wins'] = total_wins
                    data.at[data.index[i], 'trade_type'] = trade_type
                    data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed via TP'
                    data.at[data.index[i], 'exit_price'] = buy_exit_price
                    data.at[data.index[i], 'exit_date'] = exit_date
                    data.at[data.index[i], 'outcome'] = 'Win'
                    data.at[data.index[i], 'PnL'] = pnl
                    data.at[data.index[i], 'entry_price'] = buy_entry_price
                    data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                    data.at[data.index[i], 'risk'] = risk


                if data['Close'].iloc[i] < data['Open'].iloc[i] and data['Close'].iloc[i] <= data['50_MA'].iloc[i] - 1:
                    if data['High'].iloc[i] >= data['50_MA'].iloc[i] + 1 or data['High'].iloc[i-1] >= data['50_MA'].iloc[i-1] + 1:

                        for j in range(i + 1, len(data)):
                        # Check if any High before the lower Low becomes >= Upper_KC + 1
                            if data['High'].iloc[j] >= data['50_MA'].iloc[j] + 1:
                                break  # Stop the search and restart from the next suitable i

                            if data['Low'].iloc[j] <= data['Low'].iloc[i] - spread:
                                # Calculate potential PnL for this exit price
                                potential_pnl = (data['Low'].iloc[i] - spread) - buy_entry_price

                                # Check if potential PnL is 1 point or higher
                                if potential_pnl >= 1:
                                    take_profit_met = True
                                    buy_exit_price = data['Low'].iloc[i] - spread
                                    break 

                        # If the take profit condition is met and potential PnL is satisfactory, update trade details and PnL
                        if take_profit_met:
                            trade_outcome = 'Win'
                            trade_type = 'B'
                            pnl = buy_exit_price - buy_entry_price
                            cumulative_pnl += pnl
                            total_wins += 1
                            exit_date = data['Date'].iloc[i]
                            trade_open = False
                            stop_loss_moved_to_be = False

                            # Update data DataFrame with trade exit details
                            data.at[data.index[i], 'Wins'] = total_wins
                            data.at[data.index[i], 'trade_type'] = trade_type
                            data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed'
                            data.at[data.index[i], 'exit_price'] = buy_exit_price
                            data.at[data.index[i], 'exit_date'] = exit_date
                            data.at[data.index[i], 'outcome'] = trade_outcome             
                            data.at[data.index[i], 'PnL'] = pnl
                            data.at[data.index[i], 'entry_price'] = buy_entry_price
                            data.at[data.index[i], 'stop_loss'] = buy_stop_loss
                            data.at[data.index[i], 'risk'] = risk
                            continue


# Calculate cumulative PnL across the DataFrame
data['cumulative_PnL'] = data['PnL'].cumsum()

# Export the data DataFrame with trade logs to Excel
data.to_excel("MA_DAILY_RESULTS.xlsx")