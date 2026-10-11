
import pandas as pd

# Load the dataset
file_path = "data/nsl_kdd_sample.csv"
df = pd.read_csv(file_path, header=None)

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Save cleaned dataset
df.to_csv("data/cleaned_data.csv", index=False, header=False)

print("\nCleaned dataset saved successfully!")