
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
data = pd.read_csv("longfiltered_data_3m_2024.csv")

# Convert 'Datetime' column to datetime type
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d/%m/%Y %H:%M')     #, format='%d/%m/%Y %H:%M')

# Set 'Datetime' as index for the DataFrame
data.set_index('Datetime', drop=False, inplace=True)

### CALCULATING THE KELTER CHANNEL ###

# Initialize columns for Keltner Channel calculations
data['EMA'] = np.nan
data['TR'] = np.nan
data['ATR'] = np.nan
data['Upper_KC'] = np.nan
data['Lower_KC'] = np.nan

# Group data by date
grouped = data.groupby(data.index.date)

for date, group in grouped:
    # Skip calculations for the first 12 rows of each day as kelter channel period = 13
    if len(group):
        # Calculate EMA, TR, and ATR starting from the 13th row to avoid using previous day data
        ema = group['Close'].ewm(span=13, adjust=False).mean()
        tr1 = group['High'] - group['Low'][12:]
        tr2 = abs(group['High'] - group['Close'].values)
        tr3 = abs(group['Low'] - group['Close'].values)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=13).mean()

        # Calculate Upper and Lower Keltner Channels
        upper_kc = ema + 1.3 * atr
        lower_kc = ema - 1.3 * atr

        # Update the original DataFrame
        data.loc[ema.index, 'EMA'] = ema
        data.loc[tr.index, 'TR'] = tr
        data.loc[atr.index, 'ATR'] = atr
        data.loc[upper_kc.index, 'Upper_KC'] = upper_kc
        data.loc[lower_kc.index, 'Lower_KC'] = lower_kc

### DISPLAYING THE PRICE BAR CHART AND KELTER CHANNEL LINES ###

        fig = go.Figure(data=[go.Candlestick(x=group.index,
                                             open=group['Open'],
                                             high=group['High'],
                                             low=group['Low'],
                                             close=group['Close'],
                                             name="Candlesticks"),
                              go.Scatter(x=ema.index, y=upper_kc, mode='lines', name='Upper KC', line=dict(color='red')),
                              go.Scatter(x=ema.index, y=ema, mode='lines', name='EMA', line=dict(color='green')),
                              go.Scatter(x=ema.index, y=lower_kc, mode='lines', name='Lower KC', line=dict(color='blue'))])

        fig.update_layout(title=f'DJIA - Keltner Channels for {date}',
                          xaxis_title='Time',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)
        #fig.show()


# Convert 'Date' to datetime if not already
data['Date'] = pd.to_datetime(data['Date'])

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


### CALCULATING THE BACKTESTING LOGIC ###

# Creating columns
data['trade_open'] = None
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

# Initialize variables 
trade_open = False
total_losses = 0  # Counter for total losses
daily_losses = 0  # Counter for daily losses
total_wins = 0  # Counter for total wins
exit_time = None
cumulative_pnl = 0 
previous_date = None 
stop_loss_moved_to_be = False
last_daily_atr = 0

# Trade strategy variables
starting_balance = 100000
per_pt = 20
max_risk = 50
spread = 3
eod = "15:57:00"
daily_atr_divider = 10
intraday_atr_multiplier = 1.25
stop_loss_to_BE = 30
atr_stop_loss_multiplier = 2

# Iterate through the DataFrame to identify buy signals and manage trades
for i in range(len(data)):
    current_date = data['Datetime'].iloc[i].date()
    try:
        current_datetime = str(data['Datetime'].iloc[i])
        current_time = current_datetime.split()[1]
    except (IndexError, AttributeError):
        continue

    # Reset daily losses counter if a new day starts
    if current_date != previous_date:
        daily_losses = 0
        previous_date = current_date

        if i > 0:
            prev_day_last_row = data[data['Datetime'].dt.date == previous_date].index[-1]
            last_daily_atr = data.loc[prev_day_last_row, 'Daily_ATR']
        if last_daily_atr > 375:
            trade_open_hours = [("09:30:00", "15:57:00")]
        else:
            trade_open_hours = [("09:30:00", "10:00:00"), ("10:03:00", "15:57:00")]

    # Check if the current time is within the allowed trade opening hours
    within_trade_hours = False
    for start_time, end_time in trade_open_hours:
        if start_time <= current_time <= end_time:
            within_trade_hours = True
            break 

        ### OPENING TRADES LOGIC ###


    if within_trade_hours and not trade_open and daily_losses < 3:
        # Check for buy signal conditions
        if data['Close'].iloc[i] > data['Open'].iloc[i] and data['Close'].iloc[i] >= data['Lower_KC'].iloc[i] + 1:
            if data['Low'].iloc[i] <= data['Lower_KC'].iloc[i] - 1:
                # Ensure the EMA condition is met
                if abs(data['EMA'].iloc[i] - (data['High'].iloc[i] + spread)) >= 10:
                    fixed_take_profit = data['EMA'].iloc[i]  # Store the EMA value at the signal bar
                    for j in range(i + 1, i + 4):
                        if j >= len(data):
                            break
                        else:
                            if data['High'].iloc[j] >= data['High'].iloc[i] + spread:
                                if (data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread)) <= max_risk:
                                    # Buy signal conditions met, execute trade at index j
                                    trade_open = True
                                    stop_loss_moved_to_be = False
                                    Buy_entry_price = data['High'].iloc[i] + spread
                                    Buy_stop_loss = data['Low'].iloc[i] - spread
                                    risk = Buy_entry_price - Buy_stop_loss
                                    trade_open_index = j

                                    # Use the fixed EMA value from the signal bar as the take profit level
                                    take_profit_target = fixed_take_profit

                                    # Update data DataFrame with trade details at the specific row
                                    data.at[data.index[j], 'trade_open'] = 'Buy Trade Open'
                                    data.at[data.index[j], 'entry_price'] = Buy_entry_price
                                    data.at[data.index[j], 'stop_loss'] = Buy_stop_loss
                                    data.at[data.index[j], 'risk'] = risk
                                    data.at[data.index[j], 'take_profit_target'] = take_profit_target
                                    data.at[data.index[j], 'outcome'] = None
                                    data.at[data.index[j], 'daily_losses'] = daily_losses
                                    break  # Exit loop once trade is opened

### MANAGING OPEN TRADES ###
    if trade_open:
        pnl = 0
        if i > trade_open_index:

            # CHECK FOR STOP LOSS BEING HIT
            if data['Low'].iloc[i] <= Buy_stop_loss:
                trade_outcome = 'Loss'
                trade_open = False
                stop_loss_moved_to_be = False
                pnl = Buy_stop_loss - Buy_entry_price
                exit_time = data['Time'].iloc[i]
                cumulative_pnl += pnl
                if pnl < 0:
                    daily_losses += 1
                    total_losses += 1

                    data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed'
                    data.at[data.index[i], 'exit_price'] = Buy_stop_loss
                    data.at[data.index[i], 'outcome'] = trade_outcome
                    data.at[data.index[i], 'exit_time'] = exit_time
                    data.at[data.index[i], 'PnL'] = pnl
                    data.at[data.index[i], 'daily_losses'] = daily_losses
                    data.at[data.index[i], 'total_losses'] = total_losses
                    data.at[data.index[i], 'total_wins'] = total_wins

                if pnl == 0:
                    data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed at B/E'
                    data.at[data.index[i], 'exit_price'] = Buy_stop_loss
                    data.at[data.index[i], 'exit_time'] = exit_time
                    data.at[data.index[i], 'outcome'] = 'B/E'
                    data.at[data.index[i], 'PnL'] = pnl
                    data.at[data.index[i], 'daily_losses'] = daily_losses
                    data.at[data.index[i], 'total_losses'] = total_losses
                    data.at[data.index[i], 'total_wins'] = total_wins

            # MOVE STOP LOSS TO ENTRY (BREAK EVEN)
            if not stop_loss_moved_to_be and data['High'].iloc[i] >= Buy_entry_price + stop_loss_to_BE:  #data['ATR'].iloc[i] * atr_stop_loss_multiplier
                Buy_stop_loss = Buy_entry_price
                data.at[data.index[i], 'stop_loss'] = Buy_stop_loss
                data.at[data.index[i], 'trade_open'] = 'Stop loss moved to entry'
                stop_loss_moved_to_be = True

            # TAKE PROFIT USING PERCENTAGE OF THE DAILY ATR AS TARGET
           # if data['High'].iloc[i] >= take_profit_target:
            #    take_profit_met = True
             #   trade_open = False
              #  stop_loss_moved_to_be = False
               # trade_outcome = 'Win'
                #exit_time = data['Time'].iloc[i]
                #buy_exit_price = take_profit_target 
                #pnl = buy_exit_price - Buy_entry_price
                #cumulative_pnl += pnl
                #total_wins += 1

                # Update data DataFrame with trade exit details
                #data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed via TP'
                #data.at[data.index[i], 'exit_price'] = buy_exit_price
                #data.at[data.index[i], 'exit_time'] = exit_time
                #data.at[data.index[i], 'outcome'] = 'Win'
                #data.at[data.index[i], 'PnL'] = pnl
                #data.at[data.index[i], 'entry_price'] = Buy_entry_price
                #data.at[data.index[i], 'stop_loss'] = Buy_stop_loss
                #data.at[data.index[i], 'risk'] = risk
                #data.at[data.index[i], 'total_losses'] = total_losses
                #data.at[data.index[i], 'total_wins'] = total_wins


            if data['High'].iloc[i] >= take_profit_target:
                # Take-profit condition met
                trade_outcome = 'Win'
                pnl = take_profit_target - Buy_entry_price
                trade_open = False
                exit_time = data['Time'].iloc[i]
                cumulative_pnl += pnl
                total_wins += 1

                # Update data DataFrame with trade exit details
                data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed via TP'
                data.at[data.index[i], 'exit_price'] = take_profit_target
                data.at[data.index[i], 'exit_time'] = exit_time
                data.at[data.index[i], 'outcome'] = trade_outcome
                data.at[data.index[i], 'PnL'] = pnl
                data.at[data.index[i], 'entry_price'] = Buy_entry_price
                data.at[data.index[i], 'stop_loss'] = Buy_stop_loss
                data.at[data.index[i], 'risk'] = risk
                data.at[data.index[i], 'total_losses'] = total_losses
                data.at[data.index[i], 'total_wins'] = total_wins


            # if data['High'].iloc[i] >= data['Upper_KC'].iloc[trade_open_index]:
            #     # Take-profit condition met based on Upper KC
            #     trade_outcome = 'Win'
            #     pnl = data['Upper_KC'].iloc[trade_open_index] - Buy_entry_price
            #     trade_open = False
            #     exit_time = data['Time'].iloc[i]
            #     cumulative_pnl += pnl
            #     total_wins += 1

            #     # Update data DataFrame with trade exit details
            #     data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed via TP (Upper KC)'
            #     data.at[data.index[i], 'exit_price'] = data['Upper_KC'].iloc[trade_open_index]
            #     data.at[data.index[i], 'exit_time'] = exit_time
            #     data.at[data.index[i], 'outcome'] = trade_outcome
            #     data.at[data.index[i], 'PnL'] = pnl
            #     data.at[data.index[i], 'entry_price'] = Buy_entry_price
            #     data.at[data.index[i], 'stop_loss'] = Buy_stop_loss
            #     data.at[data.index[i], 'risk'] = risk
            #     data.at[data.index[i], 'total_losses'] = total_losses
            #     data.at[data.index[i], 'total_wins'] = total_wins


            # # TAKE PROFIT CONDITION MET BASED UPON BAR CLOSING BACK BELOW THE UPPER KC LEVEL AFTER PASSING THROUGH IT
            # elif data['Close'].iloc[i] < data['Open'].iloc[i] and data['Close'].iloc[i] <= data['Upper_KC'].iloc[i] - 1:
            #    if data['High'].iloc[i] >= data['Upper_KC'].iloc[i] + 1 or data['High'].iloc[i-1] >= data['Upper_KC'].iloc[i-1] + 1:
            #        take_profit_met = False

            #        for j in range(i + 1, len(data)):
            #            if j >= len(data):
            #                break 
            #            else:
            #                if data['High'].iloc[j] >= data['Upper_KC'].iloc[j] + 1:
            #                    break 

            #                if data['Low'].iloc[j] <= data['Low'].iloc[i] - spread:
            #                    # Calculate potential PnL for this exit price
            #                    potential_pnl = (data['Low'].iloc[i] - spread) - Buy_entry_price

            #                    if potential_pnl >= 1:
            #                        take_profit_met = True
            #                        Buy_exit_price = data['Low'].iloc[i] - spread
            #                        break 

            #            if take_profit_met:
            #                trade_outcome = 'Win'
            #                pnl = Buy_exit_price - Buy_entry_price
            #                cumulative_pnl += pnl
            #                total_wins += 1
            #                trade_open = False
            #                exit_time = data['Time'].iloc[i]
            #                stop_loss_moved_to_be = False

            #                # Update data DataFrame with trade exit details
            #                data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed'
            #                data.at[data.index[i], 'exit_price'] = Buy_exit_price
            #                data.at[data.index[i], 'exit_time'] = exit_time
            #                data.at[data.index[i], 'outcome'] = trade_outcome
            #                data.at[data.index[i], 'PnL'] = pnl
            #                data.at[data.index[i], 'total_losses'] = total_losses
            #                data.at[data.index[i], 'total_wins'] = total_wins

            # CLOSE TRADE AT EOD FOR PROFIT OR LOSS - NO OVERNIGHT POSITIONS
            if data['Time'].iloc[i] == eod:
                Buy_exit_price = data['Close'].iloc[i]

                # Check if the stop loss should trigger
                if data['Low'].iloc[i] <= Buy_stop_loss:
                    Buy_exit_price = Buy_stop_loss

                pnl = Buy_exit_price - Buy_entry_price  
                trade_open = False
                exit_time = data['Time'].iloc[i]
                stop_loss_moved_to_be = False
                cumulative_pnl += pnl
                if pnl >= 0:
                    trade_outcome = 'Win'
                    total_wins += 1
                if pnl < 0:
                    trade_outcome = 'Loss'
                    daily_losses += 1
                    total_losses += 1

                data.at[data.index[i], 'daily_losses'] = daily_losses
                data.at[data.index[i], 'trade_open'] = 'Buy Trade Closed at EOD'
                data.at[data.index[i], 'exit_time'] = exit_time
                data.at[data.index[i], 'exit_price'] = Buy_exit_price
                data.at[data.index[i], 'outcome'] = trade_outcome
                data.at[data.index[i], 'PnL'] = pnl
                data.at[data.index[i], 'total_losses'] = total_losses
                data.at[data.index[i], 'total_wins'] = total_wins
                continue 

### CALCULATING PNL AND STRATEGY RESULTS ###

# Calculate cumulative PnL across the DataFrame
data['cumulative_PnL'] = data['PnL'].cumsum()

plt.figure(figsize=(10, 5))
plt.plot(data['Datetime'], data['cumulative_PnL'], label='Cumulative PnL', color='green')
plt.title('Cumulative Profit and Loss over Time')
plt.xlabel('Datetime')
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
data.drop(['TR', 'ATR', 'Datetime'], axis=1, inplace=True)

### EXPORT STRATEGY RESULTS AND DATAFRAME TO EXCEL ###

# Export the data DataFrame with trade logs to Excel
data.to_excel("KC_Results_Buy.xlsx")