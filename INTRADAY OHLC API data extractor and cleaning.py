import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# Step 1: Download market data for specified months and save as XLSX (Updated Script 1)
def download_market_data():
    base_file_name = "DJIA_Data"
    file_extension = ".xlsx"
    counter = 1
    start_month = datetime(2024, 6, 1)
    end_month = datetime(2024, 8, 31)
    base_api_url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=DIA&interval=1min&extended_hours=False&outputsize=full&datatype=json&apikey=RUAB241ATPLNS4DD"

    current_month = start_month
    while current_month <= end_month:
        month_parameter = current_month.strftime("%Y-%m")
        api_url = f"{base_api_url}&month={month_parameter}"

        response = requests.get(api_url)

        if response.status_code == 200:
            api_data = response.json()
            time_series_data = api_data.get("Time Series (1min)")  # 1-min interval

            if time_series_data:
                df = pd.DataFrame(time_series_data).T.reset_index()
                df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

                # Multiply values by 100
                columns_to_multiply = ["Open", "High", "Low", "Close"]
                df[columns_to_multiply] = df[columns_to_multiply].astype(float) * 100

                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df["Date"] = df["Datetime"].dt.date
                df["Time"] = df["Datetime"].dt.time

                df = df[["Datetime", "Date", "Time", "Open", "High", "Low", "Close", "Volume"]]
                
                current_file_name = f"{base_file_name}_{month_parameter}_{counter}{file_extension}"
                df.to_excel(current_file_name, index=False, engine="openpyxl")

                print(f"Data for {month_parameter} has been saved to {current_file_name}")
                counter += 1
            else:
                print(f"No data returned for {month_parameter}")
        else:
            print(f"Error fetching data for {month_parameter}. Status: {response.status_code}")

        # Move to the next month
        current_month += timedelta(days=32)
        current_month = datetime(current_month.year, current_month.month, 1)

# Step 2: Combine monthly files into one (Previously Script 2)
def combine_monthly_files(directory_path):
    combined_df = pd.DataFrame()

    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory_path, filename)
            try:
                df = pd.read_excel(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    combined_df.to_excel(os.path.join(directory_path, "DJIA_Data_Combined.xlsx"), index=False)
    print("Monthly files combined into DJIA_Data_Combined.xlsx.")

# Step 3: Filter out unnecessary time intervals (Previously Script 3)
def filter_time_intervals(file_path):
    df = pd.read_excel(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Time'] = df['Datetime'].dt.time

    # Define the time ranges to filter out
    time_range1_start = pd.to_datetime('04:00:00').time()
    time_range1_end = pd.to_datetime('09:29:00').time()
    time_range2_start = pd.to_datetime('16:00:00').time()
    time_range2_end = pd.to_datetime('20:00:00').time()

    # Filter out unwanted time intervals
    filtered_df = df[~((df['Time'] >= time_range1_start) & (df['Time'] <= time_range1_end) |
                       (df['Time'] >= time_range2_start) & (df['Time'] <= time_range2_end))]

    sorted_df = filtered_df.sort_values(by=['Datetime'], ascending=True)
    sorted_df.to_excel('DJIA_Data_Filtered.xlsx', index=False)
    print("Time intervals filtered and saved to DJIA_Data_Filtered.xlsx.")

# Step 4: Resample the data into 3-minute intervals and save to specified location (Previously Script 4)
def resample_to_3_min_intervals():
    df = pd.read_excel("DJIA_Data_Filtered.xlsx")
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Resample into 3-minute intervals
    resampled_df = df.resample('3T', on='Datetime').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna().reset_index()

    resampled_df['Date'] = resampled_df['Datetime'].dt.date
    resampled_df['Time'] = resampled_df['Datetime'].dt.time

    # Reorder and select columns for output
    columns = ['Datetime', 'Date', 'Time', 'Open', 'High', 'Low', 'Close']
    final_df = resampled_df[columns]

    # Specify the output directory and file path
    output_dir = r"C:\Users\benmi\Desktop\Trading View Backtest\Data"
    output_file_name = os.path.join(output_dir, "Transformed_DJIA_Data_3_Min_Interval.csv")
    
    final_df.to_csv(output_file_name, index=False)
    print(f"Data resampled and saved to {output_file_name}.")

# Main function to run all steps in sequence
def main():
    download_market_data()
    
    # Assuming the directory is where the XLSX files are saved
    directory_path = os.getcwd()
    combine_monthly_files(directory_path)
    
    filter_time_intervals('DJIA_Data_Combined.xlsx')
    
    resample_to_3_min_intervals()

if __name__ == "__main__":
    main()
