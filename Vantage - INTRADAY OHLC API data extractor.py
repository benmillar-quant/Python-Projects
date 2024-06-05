

#"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=DIA&interval=1min&extended_hours=True&month=2023-12&outputsize=full&datatype=json&apikey=RUAB241ATPLNS4DD"

#&month=2023-12

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# Specify the initial file name
base_file_name = "DJIA_Data"
file_extension = ".xlsx"
counter = 1

# Start and end months in the time range
start_month = datetime(2024, 3, 1)
end_month = datetime(2024, 3, 31)

# Define the API URL without the month parameter
base_api_url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=DIA&interval=1min&extended_hours=False&outputsize=full&datatype=json&apikey=RUAB241ATPLNS4DD"

# Loop through months
current_month = start_month
while current_month <= end_month:

    month_parameter = current_month.strftime("%Y-%m")

    api_url = f"{base_api_url}&month={month_parameter}"

    response = requests.get(api_url)

    if response.status_code == 200:
        api_data = response.json()

        time_series_data = api_data.get("Time Series (1min)")                          # edit this to the time interval you need to extract

        # Convert the data to a DataFrame
        df = pd.DataFrame(time_series_data).T.reset_index()
        df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

        # Multiply values in "Open", "High", "Low", and "Close" columns by 100
        columns_to_multiply = ["Open", "High", "Low", "Close"]
        df[columns_to_multiply] = df[columns_to_multiply].astype(float) * 100

        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["Date"] = df["Datetime"].dt.date
        df["Time"] = df["Datetime"].dt.time

        df = df[["Datetime", "Date", "Time", "Open", "High", "Low", "Close", "Volume"]]

        current_file_name = f"{base_file_name}_{month_parameter}_{counter}{file_extension}"

        # Save the DataFrame to the XLSX file
        df.to_excel(current_file_name, index=False, engine="openpyxl")

        print(f"Data for {month_parameter} has been successfully saved to {current_file_name}")

        counter += 1
    else:
        print(f"Error: Unable to fetch data from the API for {month_parameter}. Status code: {response.status_code}")

    current_month = current_month + timedelta(days=32)
    current_month = datetime(current_month.year, current_month.month, 1)
