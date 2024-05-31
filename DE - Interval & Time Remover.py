
import pandas as pd


file_path = 'DJIA_Data.xlsx'
df = pd.read_excel(file_path)

# Ensure the 'Datetime' column is in proper datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Extract time from 'Datetime'
df['Time'] = df['Datetime'].dt.time

# Define the time ranges to filter out
time_range1_start = pd.to_datetime('04:00:00').time()
time_range1_end = pd.to_datetime('09:29:00').time()
time_range2_start = pd.to_datetime('16:00:00').time()
time_range2_end = pd.to_datetime('20:00:00').time()

# Filter out rows where time is within the specified ranges
filtered_df = df[~((df['Time'] >= time_range1_start) & (df['Time'] <= time_range1_end) |
                   (df['Time'] >= time_range2_start) & (df['Time'] <= time_range2_end))]


sorted_df = filtered_df.sort_values(by=['Datetime'], ascending=True)

# Export the sorted and filtered data to a new Excel file
filtered_df.to_excel('DJIA_Data_Filtered.xlsx', index=False)

print("The sorted and filtered Excel file has been saved as 'DJIA_Data_Filtered.xlsx'.")

