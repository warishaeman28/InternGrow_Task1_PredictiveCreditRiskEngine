import pandas as pd
import numpy as np

# Make random numbers the same every time
np.random.seed(42)

# Number of customers
rows = 1000

# Create customer data
data = {
    "Age": np.random.randint(18, 65, rows),
    "Income": np.random.randint(20000, 150000, rows),
    "Loan_Amount": np.random.randint(1000, 60000, rows),
    "Credit_Score": np.random.randint(300, 850, rows),
    "Employment_Years": np.random.randint(0, 30, rows),
    "Previous_Default": np.random.randint(0, 2, rows)
}

df = pd.DataFrame(data)

# Create Credit Risk
df["Credit_Risk"] = (
    (df["Credit_Score"] < 550) |
    (df["Loan_Amount"] > 35000) |
    (df["Previous_Default"] == 1)
).astype(int)

# Save CSV
import os

file_path = os.path.join(os.path.dirname(__file__), "credit_risk_dataset.csv")
df.to_csv(file_path, index=False)

print("Dataset saved at:")
print(file_path)

print("Dataset created successfully!")
print(df.head())