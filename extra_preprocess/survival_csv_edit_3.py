import pandas as pd

# Load with all columns as strings to avoid type issues
df = pd.read_csv("data/survival_final.csv", dtype=str)

# Use 'Unnamed: 1' as actual PatientID
df["PatientID"] = df["Unnamed: 1"].str.strip().str.lower().str.replace(".", "-", regex=False)

# Set as index
df.set_index("PatientID", inplace=True)

# Convert survival and death columns
df["survival"] = pd.to_numeric(df["survival"], errors="coerce")
df["death"] = pd.to_numeric(df["death"], errors="coerce").astype("Int64")

# Drop unnecessary columns
df.drop(columns=["Unnamed: 1", "overall_survival", "status", "overallsurvival"], errors="ignore", inplace=True)

# Save cleaned version
df.to_csv("data/survival_final.csv")
print("✅ Saved cleaned survival data to 'data/survival_final.csv'")
