import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import plotly.express as px
import seaborn as sns
import umap
from lifelines import KaplanMeierFitter
from model import OmicsEncoder
from data_loader import MultiOmicsDataset
from utils import load_model, set_seed, setup_logger, compute_clustering_metrics
import config
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.preprocessing import LabelEncoder


def compute_purity(y_true, y_pred):
    contingency_matrix = pd.crosstab(pd.Series(y_true), pd.Series(y_pred))
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

def kaplan_meier_plot(survival_df, cluster_labels, output_path='outputs/km_plot.png'):
    survival_df = survival_df.copy()
    survival_df["cluster"] = cluster_labels
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(survival_df["cluster"].unique()):
        group = survival_df[survival_df["cluster"] == cluster_id]
        kmf.fit(durations=group["time"], event_observed=group["event"], label=f"Cluster {cluster_id}")
        kmf.plot_survival_function(ci_show=False)

    plt.title("Kaplan-Meier Curves by Predicted Cluster")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_embeddings(embeddings, labels, prefix, output_dir, method="tsne"):
    # Convert string labels to integers for plotting
    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)
    
    print("🧬 Subtype label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

    # Convert labels to numbers
    if method == "tsne":
        reducer_2d = TSNE(n_components=2, random_state=config.SEED)
        reducer_3d = TSNE(n_components=3, random_state=config.SEED)
    elif method == "umap":
        reducer_2d = umap.UMAP(n_components=2, random_state=config.SEED)
        reducer_3d = umap.UMAP(n_components=3, random_state=config.SEED)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 2D Plot
    emb_2d = reducer_2d.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=numeric_labels, cmap="tab10", alpha=0.7)
    plt.title(f"{prefix} {method.upper()} (2D)")
    plt.colorbar(scatter, ticks=range(len(le.classes_)), label='Subtype')
    plt.savefig(os.path.join(output_dir, f"{prefix}_{method}_2d.png"))
    plt.close()

    # 3D Plot
    emb_3d = reducer_3d.fit_transform(embeddings)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=numeric_labels, cmap="tab10", alpha=0.7)
    ax.set_title(f"{prefix} {method.upper()} (3D)")
    fig.colorbar(scatter, ax=ax, ticks=range(len(le.classes_)), label='Subtype')
    plt.savefig(os.path.join(output_dir, f"{prefix}_{method}_3d.png"))
    plt.close()

def plot_tsne(embeddings, labels=None, title="t-SNE Embeddings", output_path='outputs/tsne_plot.png', is_3d=False):
    """Plot 2D or 3D t-SNE visualization."""
    tsne = TSNE(n_components=3 if is_3d else 2, random_state=config.SEED)
    reduced = tsne.fit_transform(embeddings)

    if is_3d and px:
        fig = px.scatter_3d(
            x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
            color=labels if labels is not None else None,
            title=title
        )
        fig.write_html(output_path.replace(".png", ".html"))
    else:
        plt.figure(figsize=(8, 6))
        if labels is not None:
            sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", s=40)
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], s=10)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def evaluate():
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    logger = setup_logger(log_path=os.path.join(config.SAVE_DIR, "eval.log"))

    # Load dataset
    dataset = MultiOmicsDataset(os.path.join(config.PROCESSED_DIR, "test.npz"))
    gene = dataset.gene.numpy()
    meth = dataset.meth.numpy()
    mirna = dataset.mirna.numpy()
    time = dataset.time.numpy()
    event = dataset.event.numpy()

    survival_df = pd.DataFrame({
        "time": time,
        "event": event
    })

    # Load models
    gene_encoder = OmicsEncoder(gene.shape[1], config.HIDDEN_DIM, config.EMBEDDING_DIM).to(device)
    meth_encoder = OmicsEncoder(meth.shape[1], config.HIDDEN_DIM, config.EMBEDDING_DIM).to(device)
    mirna_encoder = OmicsEncoder(mirna.shape[1], config.HIDDEN_DIM, config.EMBEDDING_DIM).to(device)

    gene_encoder = load_model(gene_encoder, os.path.join(config.SAVE_DIR, "models/gene_encoder.pth"), device)
    meth_encoder = load_model(meth_encoder, os.path.join(config.SAVE_DIR, "models/meth_encoder.pth"), device)
    mirna_encoder = load_model(mirna_encoder, os.path.join(config.SAVE_DIR, "models/mirna_encoder.pth"), device)

    # Compute embeddings
    with torch.no_grad():
        gene_emb = gene_encoder(torch.tensor(gene).to(device)).cpu().numpy()
        meth_emb = meth_encoder(torch.tensor(meth).to(device)).cpu().numpy()
        mirna_emb = mirna_encoder(torch.tensor(mirna).to(device)).cpu().numpy()

    combined_embeddings = np.concatenate([gene_emb, meth_emb, mirna_emb], axis=1)

    # Clustering
    n_clusters = getattr(config, "N_CLUSTERS", 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.SEED)
    cluster_labels = kmeans.fit_predict(combined_embeddings)

    results = {}
    sil = silhouette_score(combined_embeddings, cluster_labels)
    results["silhouette"] = float(sil)
    logger.info(f"Silhouette Score: {sil:.4f}")

    # Subtype evaluation
    if hasattr(dataset, "subtype"):
        subtype_labels = dataset.subtype
        acc, ari, nmi = compute_clustering_metrics(subtype_labels, cluster_labels)
        purity = compute_purity(subtype_labels, cluster_labels)

        results.update({
            "Accuracy": float(acc),
            "ARI": float(ari),
            "NMI": float(nmi),
            "Purity": float(purity)
        })

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"ARI: {ari:.4f}")
        logger.info(f"NMI: {nmi:.4f}")
        logger.info(f"Purity: {purity:.4f}")

        visualize_embeddings(combined_embeddings, subtype_labels, prefix="tsne_subtype", output_dir=config.SAVE_DIR, method="tsne")
        visualize_embeddings(combined_embeddings, subtype_labels, prefix="umap_subtype", output_dir=config.SAVE_DIR, method="umap")

    else:
        logger.warning("No 'subtype' labels found. Skipping subtype metrics and visualization.")

    # Kaplan-Meier plot
    kaplan_meier_plot(survival_df, cluster_labels, os.path.join(config.SAVE_DIR, "km_plot.png"))
    
    # Log-rank test
    unique_clusters = np.unique(cluster_labels)
    if len(unique_clusters) == 2:
        group1 = (cluster_labels == unique_clusters[0])
        group2 = (cluster_labels == unique_clusters[1])
        result = logrank_test(
            survival_df["time"][group1], survival_df["time"][group2],
            event_observed_A=survival_df["event"][group1],
            event_observed_B=survival_df["event"][group2]
        )
        logrank_p = result.p_value
        results["logrank_p"] = float(logrank_p)
        logger.info(f"Log-rank test p-value: {logrank_p:.4e}")
    elif len(unique_clusters) > 2:
        result = multivariate_logrank_test(
            survival_df["time"],
            groups=cluster_labels,
            event_observed=survival_df["event"]
        )
        logrank_p = result.p_value
        results["logrank_p"] = float(logrank_p)
        logger.info(f"Multivariate Log-rank test p-value: {logrank_p:.4e}")
    else:
        logger.warning("Log-rank test skipped: fewer than 2 clusters")

    # Concordance index (based on cluster risk score)
    c_index = concordance_index(
        survival_df["time"], -cluster_labels, survival_df["event"]
    )
    results["c_index"] = float(c_index)
    logger.info(f"Concordance Index (C-index): {c_index:.4f}")

    # Cluster embedding visualizations
    visualize_embeddings(combined_embeddings, cluster_labels, prefix="tsne_cluster", output_dir=config.SAVE_DIR, method="tsne")
    visualize_embeddings(combined_embeddings, cluster_labels, prefix="umap_cluster", output_dir=config.SAVE_DIR, method="umap")

    # Save metrics
    with open(os.path.join(config.SAVE_DIR, "eval_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("✅ Evaluation complete and metrics saved.")
    
    
        # --- UMAP Visualization ---
    umap_2d = umap.UMAP(n_components=2, random_state=config.SEED)
    umap_3d = umap.UMAP(n_components=3, random_state=config.SEED)

    embedding_2d = umap_2d.fit_transform(combined_embeddings)
    embedding_3d = umap_3d.fit_transform(combined_embeddings)

    # 2D Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=cluster_labels, palette="tab10")
    plt.title("2D UMAP of Combined Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "umap_2d.png"))
    plt.close()

    # 3D Plot (HTML)
    try:
        import plotly.express as px
        fig_3d = px.scatter_3d(
            x=embedding_3d[:, 0],
            y=embedding_3d[:, 1],
            z=embedding_3d[:, 2],
            color=[str(label) for label in cluster_labels],
            title="3D UMAP of Combined Embeddings"
        )
        fig_3d.write_html(os.path.join(config.SAVE_DIR, "umap_3d.html"))
        logger.info("✅ UMAP visualizations saved (2D PNG, 3D HTML)")
    except ImportError:
        logger.warning("Plotly not installed, skipping 3D UMAP visualization.")
        
    # t-SNE Visualizations
    plot_tsne(combined_embeddings, cluster_labels, title="t-SNE Clustering (2D)", output_path=os.path.join(config.SAVE_DIR, "tsne_clusters.png"))
    if px:
        plot_tsne(combined_embeddings, cluster_labels, title="t-SNE Clustering (3D)", output_path=os.path.join(config.SAVE_DIR, "tsne_clusters_3d.png"), is_3d=True)

    if "subtype" in survival_df.columns:
        plot_tsne(combined_embeddings, survival_df["subtype"].values, title="t-SNE Subtypes (2D)", output_path=os.path.join(config.SAVE_DIR, "tsne_subtypes.png"))
        if px:
            plot_tsne(combined_embeddings, survival_df["subtype"].values, title="t-SNE Subtypes (3D)", output_path=os.path.join(config.SAVE_DIR, "tsne_subtypes_3d.png"), is_3d=True)




if __name__ == "__main__":
    evaluate()
