import pandas as pd
import numpy as np

def inspect_csv(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Print basic information about the DataFrame
    print("Basic Information:")
    print(df.info())

    # Print the first few rows of the DataFrame
    print("\nFirst few rows:")
    print(df.head())

    # Print the column names
    print("\nColumn Names:")
    print(df.columns)

def create_csv(file_path):
    # Define the structure of the DataFrame
    data = {
        'dataset': ['train', 'dev', 'test'],
        'delta_time': [np.random.uniform(0, 60) for _ in range(3)],
        'end_time': [pd.to_datetime('2019-06-07 00:13:07') for _ in range(3)],
        'interval_id': ['1', '2', 'live'],
        'speaker': ['oliver'] * 3,
        'start_time': [pd.to_datetime('2019-06-07 00:12:41') for _ in range(3)],
        'video_fn': ['video250'] * 3,
        'video_link': ['http://example.com/video1'] * 3
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a new CSV file
    df.to_csv(file_path, index=False)

if __name__ == '__main__':
    file_path = 'E:\\PATS_fake\\pats\\data\\new_file.csv'
    new_file_path = 'E:\\PATS_fake\\pats\\data\\new_file.csv'
    inspect_csv(file_path)
    # create_csv(new_file_path)
    
