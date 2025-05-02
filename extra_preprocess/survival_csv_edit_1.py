import pandas as pd

# Load the raw transposed-style survival file
raw = pd.read_csv("data/survival.csv", header=None)

# Set first column as new header
raw = raw.set_index(0).transpose()

# Convert index to string before using .str methods
raw.index = raw.index.astype(str).str.lower().str.replace(".", "-", regex=False)
raw.index.name = "PatientID"

# Save to a new cleaned CSV
raw.to_csv("data/survival_cleaned.csv")
print("✅ Transposed and cleaned survival data saved to 'data/survival_cleaned.csv'")
