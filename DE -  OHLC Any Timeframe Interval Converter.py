
import pandas as pd

# Step 1: Read the XLSX file
df = pd.read_excel("DJIA_Data_Filtered.xlsx")

# Ensure 'Datetime' column is in datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Step 2: Resample the data into 3-minute intervals
resampled_df = df.resample('3T', on='Datetime').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last'
}).dropna().reset_index()

# The 'Time' column will be extracted from the new 'Datetime' index
resampled_df['Date'] = resampled_df['Datetime'].dt.date
resampled_df['Time'] = resampled_df['Datetime'].dt.time

# Reorder and select the desired columns for the output
columns = ['Datetime', 'Date', 'Time', 'Open', 'High', 'Low', 'Close']
final_df = resampled_df[columns]

# Step 3: Write the new DataFrame to a new XLSX file
output_file_name = "Transformed_DJIA_Data_3_Min_Interval.xlsx"
final_df.to_excel(output_file_name, index=False)

print(f"Data transformed and saved to {output_file_name}")
