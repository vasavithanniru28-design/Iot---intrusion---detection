


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_data.csv", header=None)

# Last column contains the class label
X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].astype(str)

# Convert text features into numbers and save encoders
feature_encoders = {}

for col in X.select_dtypes(include=["object"]).columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col].astype(str))
    feature_encoders[col] = encoder

joblib.dump(feature_encoders, "models/feature_encoders.pkl")

# Convert labels into numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save trained model
joblib.dump(model, "models/intrusion_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nModel saved successfully!")