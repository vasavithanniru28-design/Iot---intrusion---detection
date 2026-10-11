
import pandas as pd

df = pd.read_csv("data/cleaned_data.csv", header=None)

# Remove the last column (class label)
X = df.iloc[:, :-1]

# Save a small sample for testing
X.head(20).to_csv(
    "data/test_network_data.csv",
    index=False,
    header=False
)

print("Test CSV created successfully!")
print("Feature columns:", X.shape[1])