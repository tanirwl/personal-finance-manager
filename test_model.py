import joblib
import pandas as pd

# Load your trained category classifier
model = joblib.load("models/category_classifier.joblib")

# Test custom incoming transactions
new_transactions = pd.DataFrame([
    {"description": "whole foods market groceries", "amount": 84.50},
    {"description": "shell gas station fillup", "amount": 45.00},
    {"description": "netflix monthly subscription", "amount": 15.49}
])

# Predict categories
predictions = model.predict(new_transactions)

print("=== Live Transaction Predictions ===")
for desc, cat in zip(new_transactions["description"], predictions):
    print(f"Transaction: '{desc}' --> Predicted Category: **{cat}**")