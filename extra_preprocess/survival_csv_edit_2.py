import pandas as pd

# Force all columns to be string type
df = pd.read_csv("data/survival_cleaned.csv", dtype=str)

# Normalize PatientID and set as index
df["PatientID"] = df["PatientID"].str.strip().str.lower().str.replace(".", "-", regex=False)
df.set_index("PatientID", inplace=True)

# Extract numeric parts and convert safely
df["survival"] = df["overall_survival"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
df["death"] = df["status"].str.extract(r"(\d)")[0].astype("Int64")  # ← Nullable Int type

# Drop raw columns
df.drop(columns=["overall_survival", "status", "overallsurvival"], errors="ignore", inplace=True)

# Save final cleaned survival file
df.to_csv("data/survival_final.csv")
print("✅ Cleaned and saved to 'data/survival_final.csv'")
