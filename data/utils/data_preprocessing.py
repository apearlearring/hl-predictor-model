import pandas as pd


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by validating, cleaning, and normalizing it.

    :param data: The raw input data
    :return: Cleaned and validated data ready for training
    :raises: ValueError if validation fails
    """
    required_columns = ['time', 'current_price', 'funding', 'open_interest', 'premium', 
                    'day_ntl_vlm',  'long_number', 'short_number']

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Drop rows with missing values in required columns
    data = data.dropna(subset=required_columns)
    # print sample

    # Optionally: You can normalize or scale the data here if needed
    # Example: data[required_columns] = (data[required_columns] - data[required_columns].mean()) / data[required_columns].std()

    print("Data validation and preprocessing completed successfully.")
    return data[required_columns]
