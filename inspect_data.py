
import pandas as pd
import os

file_path = "data/nsl_kdd_sample.csv"

if not os.path.exists(file_path):
    print("Dataset file not found!")
else:
    print("Dataset file found!")

    try:
        df = pd.read_csv(file_path, header=None)

        print("\nDataset loaded successfully!")
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nColumn data types:")
        print(df.dtypes)

        print("\nMissing values:")
        print(df.isnull().sum().sum())

    except Exception as e:
        print("Error:", e)