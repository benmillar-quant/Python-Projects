import requests
import pandas as pd

base_api_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=DIA&outputsize=full&datatype=json&apikey=RUAB241ATPLNS4DD"


response = requests.get(base_api_url)

if response.status_code == 200:
    data = response.json()
    
    # Extract the daily time series data
    daily_data = data.get("Time Series (Daily)", {})
    
    # Prepare data for DataFrame, multiplying values by 100
    formatted_data = []
    for date, daily_info in daily_data.items():
        formatted_data.append({
            "Date": date,
            "Open": float(daily_info["1. open"]) * 100,
            "High": float(daily_info["2. high"]) * 100,
            "Low": float(daily_info["3. low"]) * 100,
            "Close": float(daily_info["4. close"]) * 100
        })
    
    # Create DataFrame
    df = pd.DataFrame(formatted_data)
    

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', ascending=True, inplace=True)
    

    df.reset_index(drop=True, inplace=True)
    
    # Export DataFrame to CSV
    df.to_csv("DJIA_DAILY_DATA.csv", index=False)
    print("Data exported to DJIA_DAILY_DATA.csv successfully.")
else:
    print(f"Failed to fetch data, status code: {response.status_code}")