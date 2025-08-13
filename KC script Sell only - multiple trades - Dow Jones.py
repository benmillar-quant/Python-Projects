import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
data = pd.read_csv("shortfiltered_data_3m_2024.csv")

# Convert 'Datetime' column to datetime type
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d/%m/%Y %H:%M')         #, format='%d/%m/%Y %H:%M')

# Set 'Datetime' as index for the DataFrame
data.set_index('Datetime', drop=False, inplace=True)

### CALCULATING THE KELTNER CHANNEL ###

# Initialize columns for Keltner Channel calculations
data['EMA'] = np.nan
data['TR'] = np.nan
data['ATR'] = np.nan
data['Upper_KC'] = np.nan
data['Lower_KC'] = np.nan

# Group data by date
grouped = data.groupby(data.index.date)

for date, group in grouped:
    if len(group):
        ema = group['Close'].ewm(span=13, adjust=False).mean()
        tr1 = group['High'] - group['Low'][12:]
        tr2 = abs(group['High'] - group['Close'].values)
        tr3 = abs(group['Low'] - group['Close'].values)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=13).mean()

        upper_kc = ema + 1.3 * atr
        lower_kc = ema - 1.3 * atr

        data.loc[ema.index, 'EMA'] = ema
        data.loc[tr.index, 'TR'] = tr
        data.loc[atr.index, 'ATR'] = atr
        data.loc[upper_kc.index, 'Upper_KC'] = upper_kc
        data.loc[lower_kc.index, 'Lower_KC'] = lower_kc

### DISPLAYING THE PRICE BAR CHART AND KELTNER CHANNEL LINES ###

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

daily_high_low = data.groupby(data['Date'].dt.date).agg(Daily_High=('High', 'max'), Daily_Low=('Low', 'min')).reset_index()
daily_high_low['Date'] = pd.to_datetime(daily_high_low['Date'])
daily_high_low['Daily_Range'] = daily_high_low['Daily_High'] - daily_high_low['Daily_Low']

daily_high_low['Daily_ATR'] = daily_high_low['Daily_Range'].rolling(window=5).mean()

data = pd.merge(data, daily_high_low[['Date', 'Daily_ATR']], on='Date', how='left')

data['Daily_ATR'] = data.groupby(data['Date'].dt.date)['Daily_ATR'].shift(1)

### CALCULATING THE BACKTESTING LOGIC ###

# Creating columns for recording multiple trades
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

# Initialize variables
total_losses = 0
daily_losses = 0
total_wins = 0
cumulative_pnl = 0
previous_date = None
last_daily_atr = 0

# Trade strategy variables
starting_balance = 100000
per_pt = 20
max_risk = 80
spread = 3
eod = "15:57:00"
daily_atr_divider = 10
intraday_atr_multiplier = 1.25
stop_loss_to_BE = 30

# Creating a List to hold active trades
active_trades = []

# Iterate through the DataFrame to identify sell signals and manage trades
for i in range(len(data)):
    current_date = data['Datetime'].iloc[i].date()
    try:
        current_datetime = str(data['Datetime'].iloc[i])
        current_time = current_datetime.split()[1]
    except (IndexError, AttributeError):
        continue

    if current_date != previous_date:
        daily_losses = 0
        previous_date = current_date

        # Close all trades at EOD
        for trade in active_trades:
            if trade.get('active', False):
                trade['active'] = False
        active_trades = []  # Reset the list for the next day

        if i > 0:
            prev_day_last_row = data[data['Datetime'].dt.date == previous_date].index[-1]
            last_daily_atr = data.loc[prev_day_last_row, 'Daily_ATR']
        if last_daily_atr > 375:
            trade_open_hours = [("09:30:00", "15:57:00")]
        else:
            trade_open_hours = [("09:30:00", "10:00:00"), ("10:03:00", "15:57:00")]

    within_trade_hours = False
    for start_time, end_time in trade_open_hours:
        if start_time <= current_time <= end_time:
            within_trade_hours = True
            break

    if daily_losses < 3:
        # Allow new trades only if previous trades have moved their stop loss to breakeven
        can_open_new_trade = True
        for trade in active_trades:
            if trade['active'] and not trade['stop_loss_moved_to_be']:
                can_open_new_trade = False
                break

        if within_trade_hours and can_open_new_trade:
            if data['Close'].iloc[i] < data['Open'].iloc[i] and data['Close'].iloc[i] <= data['Upper_KC'].iloc[i] - 1:
                if data['High'].iloc[i] >= data['Upper_KC'].iloc[i] + 1:
                    for j in range(i+1, i+4):
                        if j >= len(data):
                            break 
                        else:
                            if data['Low'].iloc[j] <= data['Low'].iloc[i] - spread:
                                if (data['High'].iloc[i] + spread - (data['Low'].iloc[i] - spread)) <= max_risk:
                                    trade_details = {
                                        'active': True,
                                        'trade_type': 'Sell',
                                        'entry_price': data['Low'].iloc[i] - spread,
                                        'stop_loss': data['High'].iloc[i] + spread,
                                        'risk': (data['High'].iloc[i] + spread) - data['Low'].iloc[i] + spread,
                                        'take_profit_target': data['Low'].iloc[i] - spread - (last_daily_atr / daily_atr_divider),
                                        'trade_open_idx': j,
                                        'stop_loss_moved_to_be': False,
                                    }
                                    active_trades.append(trade_details)

                                    # Record the trade in the DataFrame at index j
                                    data.at[data.index[j], 'status'] = 'Sell Trade Opened'
                                    data.at[data.index[j], 'entry_price'] = trade_details['entry_price']
                                    data.at[data.index[j], 'stop_loss'] = trade_details['stop_loss']
                                    data.at[data.index[j], 'risk'] = trade_details['risk']
                                    data.at[data.index[j], 'take_profit_target'] = trade_details['take_profit_target']
                                    break

    for trade in active_trades:
        if trade['active']:
            trade_open_index = trade['trade_open_idx']

            if i > trade_open_index:
                if data['High'].iloc[i] >= trade['stop_loss']:
                    trade['active'] = False
                    pnl = trade['entry_price'] - trade['stop_loss']
                    cumulative_pnl += pnl
                    if pnl < 0:
                        daily_losses += 1
                        total_losses += 1

                    outcome = 'Loss' if pnl < 0 else 'B/E'

                    data.at[data.index[trade_open_index], 'status'] = 'Sell Trade Closed'
                    data.at[data.index[trade_open_index], 'exit_price'] = trade['stop_loss']
                    data.at[data.index[trade_open_index], 'exit_time'] = data['Time'].iloc[i]
                    data.at[data.index[trade_open_index], 'outcome'] = outcome
                    data.at[data.index[trade_open_index], 'PnL'] = pnl
                    data.at[data.index[trade_open_index], 'daily_losses'] = daily_losses
                    data.at[data.index[trade_open_index], 'total_losses'] = total_losses
                    data.at[data.index[trade_open_index], 'total_wins'] = total_wins

                if not trade['stop_loss_moved_to_be'] and data['Low'].iloc[i] <= trade['entry_price'] - stop_loss_to_BE:
                    trade['stop_loss'] = trade['entry_price']
                    trade['stop_loss_moved_to_be'] = True
                    data.at[data.index[trade_open_index], 'stop_loss'] = trade['entry_price']

#                if data['Low'].iloc[i] <= trade['take_profit_target']:
 #                   trade['active'] = False
  #                  pnl = trade['entry_price'] - trade['take_profit_target']
   #                 cumulative_pnl += pnl
    #                total_wins += 1
#
 #                   data.at[data.index[trade_open_index], 'status'] = 'Sell Trade Closed via TP'
  #                  data.at[data.index[trade_open_index], 'exit_price'] = trade['take_profit_target']
   #                 data.at[data.index[trade_open_index], 'exit_time'] = data['Time'].iloc[i]
    #                data.at[data.index[trade_open_index], 'outcome'] = 'Win'
     #               data.at[data.index[trade_open_index], 'PnL'] = pnl
      #              data.at[data.index[trade_open_index], 'total_losses'] = total_losses
       #             data.at[data.index[trade_open_index], 'total_wins'] = total_wins

#                elif data['Close'].iloc[i] > data['Open'].iloc[i] and data['Close'].iloc[i] >= data['Lower_KC'].iloc[i] + 1:
 #                   if data['Low'].iloc[i] <= data['Lower_KC'].iloc[i] - 1 or data['Low'].iloc[i-1] <= data['Lower_KC'].iloc[i-1] - 1:
  #                      take_profit_met = False

   #                     for j in range(i + 1, len(data)):
    #                        if j >= len(data):
     #                           break 
      #                      else:
       #                         if data['Low'].iloc[j] <= data['Lower_KC'].iloc[j] - 1:
        #                            break 

         #                       if data['High'].iloc[j] >= data['High'].iloc[i] + spread:
          #                          potential_pnl = trade['entry_price'] - (data['High'].iloc[i] + spread)

           #                         if potential_pnl >= 1:
            #                            take_profit_met = True
             #                           trade['exit_price'] = data['High'].iloc[i] + spread
              #                          break 

               #         if take_profit_met:
                #            trade['active'] = False
                 #           pnl = trade['entry_price'] - trade['exit_price']
                  #          cumulative_pnl += pnl
                   #         total_wins += 1
#
 #                           data.at[data.index[trade_open_index], 'status'] = 'Sell Trade Closed'
  #                          data.at[data.index[trade_open_index], 'exit_price'] = trade['exit_price']
   #                         data.at[data.index[trade_open_index], 'exit_time'] = data['Time'].iloc[i]
    #                        data.at[data.index[trade_open_index], 'outcome'] = 'Win'
     #                       data.at[data.index[trade_open_index], 'PnL'] = pnl
      #                      data.at[data.index[trade_open_index], 'total_losses'] = total_losses
       #                     data.at[data.index[trade_open_index], 'total_wins'] = total_wins

                if data['Time'].iloc[i] == eod:
                    trade['active'] = False
                    exit_price = data['Close'].iloc[i]
                    if data['High'].iloc[i] >= trade['stop_loss']:
                        exit_price = trade['stop_loss']

                    pnl = trade['entry_price'] - exit_price
                    cumulative_pnl += pnl
                    outcome = 'Win' if pnl > 0 else 'Loss' if pnl < 0 else 'B/E'

                    if pnl >= 0:
                        total_wins += 1
                    else:
                        daily_losses += 1
                        total_losses += 1

                    data.at[data.index[trade_open_index], 'status'] = 'Sell Trade Closed at EOD'
                    data.at[data.index[trade_open_index], 'exit_price'] = exit_price
                    data.at[data.index[trade_open_index], 'exit_time'] = data['Time'].iloc[i]
                    data.at[data.index[trade_open_index], 'outcome'] = outcome
                    data.at[data.index[trade_open_index], 'PnL'] = pnl
                    data.at[data.index[trade_open_index], 'daily_losses'] = daily_losses
                    data.at[data.index[trade_open_index], 'total_losses'] = total_losses
                    data.at[data.index[trade_open_index], 'total_wins'] = total_wins

### CALCULATING PNL AND STRATEGY RESULTS ###

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
print("Starting balance: £", starting_balance)
print("Cash returns @ £{} per pt: £{:.2f}".format(per_pt, cumulative_pnl * per_pt))

cash_returns = cumulative_pnl * per_pt
final_balance = starting_balance + cash_returns
percentage_returns = ((final_balance - starting_balance) / starting_balance) * 100

print("Percentage returns: {:.2f}%".format(percentage_returns))

# Drop unnecessary columns
data.drop(['EMA', 'TR', 'ATR', 'Datetime', 'Daily_ATR', 'trade_type', 'take_profit_target'], axis=1, inplace=True)

### EXPORT STRATEGY RESULTS AND DATAFRAME TO EXCEL ###

data.to_excel("KC_Results_Sell.xlsx")
