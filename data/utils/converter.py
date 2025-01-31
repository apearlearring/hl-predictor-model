import pandas as pd

def json_to_csv_converter():
    # Read JSON data
    json_data = pd.read_json("data/sets/BTC_metrics.json")
    
    # Assuming 'metrics' contains the nested data, normalize it into a dataframe
    df = pd.json_normalize(json_data['metrics'])
    
    # Ensure all required columns are present and in the correct order
    columns = [
        'time',
        'coin',
        'funding',
        'open_interest',
        'premium',
        'day_ntl_vlm',
        'current_price',
        'long_number',
        'short_number'
    ]
    
    # Reorder columns (will only include columns that exist in the data)
    df = df[columns]
    
    # Save to CSV
    df.to_csv("data/sets/BTC_metrics.csv", index=False)
    
    print("Conversion completed successfully.")

if __name__ == "__main__":
    json_to_csv_converter()
    