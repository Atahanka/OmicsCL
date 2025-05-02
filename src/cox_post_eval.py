import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

print("🔍 Evaluating C-index for multiple cluster sizes...\n")

# Load saved modality embeddings
gene = np.load("./outputs/embeddings/gene_embeddings.npy")
meth = np.load("./outputs/embeddings/meth_embeddings.npy")
mirna = np.load("./outputs/embeddings/mirna_embeddings.npy")

# Concatenate across omics views
combined_emb = np.concatenate([gene, meth, mirna], axis=1)

# Load survival labels from val set
val = np.load("./processed/val.npz")
time = val["time"]
event = val["event"]

results = []

for k in [2, 3, 4, 5, 6, 7, 8, 9]:
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(combined_emb)

        # Prepare dataframe
        df = pd.DataFrame({
            "time": time,
            "event": event,
            "cluster": pd.Categorical(cluster_labels)
        })

        # Fit Cox model using categorical cluster variable
        cox = CoxPHFitter()
        cox.fit(df, duration_col="time", event_col="event", formula="cluster")

        # Compute concordance index
        c_index = concordance_index(df["time"], -cox.predict_partial_hazard(df), df["event"])
        results.append((k, c_index))
        print(f"✅ k={k:2d} | C-index: {c_index:.4f}")

    except Exception as e:
        print(f"❌ k={k:2d} | Failed: {e}")

# Save results
df_res = pd.DataFrame(results, columns=["n_clusters", "c_index"])
os.makedirs("./outputs", exist_ok=True)
df_res.to_csv("./outputs/c_index_vs_k.csv", index=False)
print("\n📈 Saved C-index results to: ./outputs/c_index_vs_k.csv")
