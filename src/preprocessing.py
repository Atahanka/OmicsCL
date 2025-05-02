import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Paths
data_dir = "data"
output_dir = "processed"
os.makedirs(output_dir, exist_ok=True)

# Load and transpose omics data (columns = samples)
gene = pd.read_csv(f"{data_dir}/gene_expression.csv", index_col=0)
meth = pd.read_csv(f"{data_dir}/dna_methylation.csv", index_col=0)
mirna = pd.read_csv(f"{data_dir}/mirna_expression.csv", index_col=0)
surv = pd.read_csv(f"{data_dir}/survival_final.csv", index_col=0)

# Updated normalize function
def normalize_ids(idx, strip_suffix=False):
    ids = idx.astype(str).str.lower().str.replace('"', '').str.replace('.', '-', regex=False)
    if strip_suffix:
        ids = ids.str.replace(r"-\d\d$", "", regex=True)
    return ids

# Normalize sample IDs
gene.index = normalize_ids(gene.index.to_series(), strip_suffix=True)
meth.index = normalize_ids(meth.index.to_series(), strip_suffix=True)
mirna.index = normalize_ids(mirna.index.to_series(), strip_suffix=True)
surv.index = normalize_ids(surv.index.to_series())


# Print debug info
print("🔍 Sample ID Counts:")
print(f"Gene:        {len(gene)}")
print(f"Methylation: {len(meth)}")
print(f"miRNA:       {len(mirna)}")
print(f"Survival:    {len(surv)}")

print("\n👀 Gene IDs:", list(gene.index[:5]))
print("👀 Methylation IDs:", list(meth.index[:5]))
print("👀 miRNA IDs:", list(mirna.index[:5]))
print("👀 Survival IDs:", list(surv.index[:5]))

# Find common samples
common = set(gene.index) & set(meth.index) & set(mirna.index) & set(surv.index)
print("\n🔁 Overlap Checks:")
print("Gene ∩ Survival:", len(set(gene.index) & set(surv.index)))
print("Methylation ∩ Survival:", len(set(meth.index) & set(surv.index)))
print("miRNA ∩ Survival:", len(set(mirna.index) & set(surv.index)))
print("All ∩ Survival:", len(common))

if len(common) == 0:
    raise ValueError("❌ No common samples found across all omics and survival data!")

# Filter and sort
common = sorted(common)
gene = gene.loc[common]
meth = meth.loc[common]
mirna = mirna.loc[common]
surv = surv.loc[common]

# Z-score normalization
scaler = StandardScaler()
gene = scaler.fit_transform(gene)
meth = scaler.fit_transform(meth)
mirna = scaler.fit_transform(mirna)

gene = np.nan_to_num(gene)
meth = np.nan_to_num(meth)
mirna = np.nan_to_num(mirna)

# Drop rows with missing survival info
print("🧪 Dropping rows with NaN in survival time or death...")
print("   → Before:", len(surv))
print("   → NaNs in survival:", surv["survival"].isna().sum())
print("   → NaNs in death:   ", surv["death"].isna().sum())

surv = surv.dropna(subset=["survival", "death"])

# Recompute common index after filtering survival
common = sorted(set(gene_index := set(common)) & set(surv.index))
gene = gene[[i in common for i in gene_index]]
meth = meth[[i in common for i in gene_index]]
mirna = mirna[[i in common for i in gene_index]]
surv = surv.loc[common]

# Final survival arrays
time = surv["survival"].astype(float).values
event = surv["death"].astype(int).values

# Also extract subtype labels (e.g., pam50) for evaluation, if available
subtypes = surv["pam50"].fillna("Unknown").values

# Split data
indices = np.arange(len(common))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# Save .npz files
def save_split(name, idx):
    np.savez_compressed(
        f"{output_dir}/{name}.npz",
        gene=gene[idx],
        meth=meth[idx],
        mirna=mirna[idx],
        time=time[idx],
        event=event[idx],
        subtype=subtypes[idx]
    )

save_split("train", train_idx)
save_split("val", val_idx)
save_split("test", test_idx)

print("\n✅ Data preprocessing complete.")
print(f"Train samples: {len(train_idx)}")
print(f"Val samples:   {len(val_idx)}")
print(f"Test samples:  {len(test_idx)}")

print("All ∩ Survival:", len(common))
