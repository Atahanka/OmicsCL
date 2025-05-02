import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from lifelines.utils import concordance_index

from model import OmicsEncoder
from contrastive_loss import nt_xent_loss, survival_contrastive_loss
from utils import save_model, save_embeddings, set_seed, setup_logger
from data_loader import get_dataloaders
import config

import matplotlib.pyplot as plt
import numpy as np
import math

def compute_hist_data(times, events, bins=20):
    times = np.array(times)
    events = np.array(events)
    deceased = times[events == 1]
    censored = times[events == 0]
    hist_d, _ = np.histogram(deceased, bins=bins)
    hist_c, bin_edges = np.histogram(censored, bins=bins)
    return hist_d, hist_c, bin_edges

def plot_combined_survival_histograms(histograms, save_path):
    num_epochs = len(histograms)
    cols = 5
    rows = math.ceil(num_epochs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, (hist_d, hist_c, bin_edges) in enumerate(histograms):
        ax = axes[i]
        width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_edges[:-1], hist_d, width=width, alpha=0.7, label="Deceased")
        ax.bar(bin_edges[:-1], hist_c, width=width, alpha=0.7, label="Censored", bottom=hist_d)
        ax.set_title(f"Epoch {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train():
    # Setup
    logger = setup_logger(log_path=os.path.join(config.SAVE_DIR, "training.log"))
    device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device}")
    set_seed(config.SEED)
    
    epoch_histograms = []

    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config.PROCESSED_DIR,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    # Infer input dimensions
    for batch in train_loader:
        input_dims = {
            'gene': batch["gene"].shape[1],
            'meth': batch["meth"].shape[1],
            'mirna': batch["mirna"].shape[1]
        }
        break

    # Initialize encoders
    omics_encoders = nn.ModuleDict({
        omic: OmicsEncoder(input_dim=dim, hidden_dim=config.HIDDEN_DIM, projection_dim=config.EMBEDDING_DIM).to(device)
        for omic, dim in input_dims.items()
    })

    all_params = [p for encoder in omics_encoders.values() for p in encoder.parameters()]
    optimizer = optim.Adam(all_params, lr=config.MIN_LR, weight_decay=config.WEIGHT_DECAY)

    scheduler = CyclicLR(
        optimizer,
        base_lr=config.MIN_LR,
        max_lr=config.MAX_LR,
        step_size_up=config.STEP_SIZE_UP,
        mode=config.CLR_MODE,
        cycle_momentum=config.CYCLE_MOMENTUM
    )


    omics_keys = list(input_dims.keys())
    logger.info(f"Training with omics views: {omics_keys}")
    logger.info(f"Epochs: {config.EPOCHS}, Batch Size: {config.BATCH_SIZE}, Cyclic LR: ({config.MIN_LR} → {config.MAX_LR})")

    # Early stopping
    best_cindex = -1.0
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        omics_encoders.train()

        for batch in train_loader:
            views = {omic: batch[omic].to(device) for omic in omics_keys}
            times = batch["time"].to(device)
            events = batch["event"].to(device)

            optimizer.zero_grad()

            # Get embeddings for each omics view
            embeddings = {
                omic: omics_encoders[omic](views[omic])
                for omic in omics_keys
            }

            # --- Contrastive Loss ---
            loss = 0
            num_pairs = 0
            for i in range(len(omics_keys)):
                for j in range(i + 1, len(omics_keys)):
                    z1 = embeddings[omics_keys[i]]
                    z2 = embeddings[omics_keys[j]]
                    loss += nt_xent_loss(z1, z2, temperature=config.TEMPERATURE)
                    num_pairs += 1

            loss /= num_pairs

            # --- Survival Contrastive Regularization ---
            # Stack and average omics embeddings
            stacked = torch.stack(list(embeddings.values()), dim=0)  # [num_omics, batch, dim]
            avg_embedding = stacked.mean(dim=0)  # [batch, dim]

            surv_loss = survival_contrastive_loss(avg_embedding, times, events, margin=1.0, scale=1.0)
            total_combined_loss = loss + config.SURVIVAL_LOSS_WEIGHT * surv_loss

            total_combined_loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += total_combined_loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"[Epoch {epoch+1}/{config.EPOCHS}] Avg Total Loss: {avg_loss:.4f} (NT-Xent + Survival)")

        
        hist_d, hist_c, bin_edges = compute_hist_data(times, events)
        epoch_histograms.append((hist_d, hist_c, bin_edges))

        # Validation C-index check
        omics_encoders.eval()
        all_embeddings = []
        survival_times = []
        survival_events = []

        with torch.no_grad():
            for batch in val_loader:
                combined_z = []
                for omic in omics_keys:
                    z = omics_encoders[omic](batch[omic].to(device))
                    combined_z.append(z.cpu())
                combined = torch.cat(combined_z, dim=1)
                all_embeddings.append(combined)

                survival_times.extend(batch["time"].cpu().numpy())
                survival_events.extend(batch["event"].cpu().numpy())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        risks = all_embeddings.mean(axis=1)  # simple proxy for risk

        val_cindex = concordance_index(survival_times, -risks, survival_events)
        logger.info(f"[Epoch {epoch+1}] Validation C-index: {val_cindex:.4f}")

        if val_cindex > best_cindex + 1e-4:
            best_cindex = val_cindex
            patience_counter = 0
            for omic in omics_keys:
                save_model(omics_encoders[omic], os.path.join(config.SAVE_DIR, "models", f"{omic}_encoder.pth"))
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                logger.info(f"⏹️ Early stopping triggered by C-index at epoch {epoch+1}")
                break

    logger.info("✅ Training complete!")

    # Save embeddings from val set
    embeddings_dict = {}
    labels = []
    omics_encoders.eval()

    with torch.no_grad():
        for batch in val_loader:
            for omic in omics_keys:
                X = batch[omic].to(device)
                z = omics_encoders[omic](X)
                if omic not in embeddings_dict:
                    embeddings_dict[omic] = z.cpu()
                else:
                    embeddings_dict[omic] = torch.cat([embeddings_dict[omic], z.cpu()], dim=0)

            labels.extend(batch["event"].numpy())

    for omic in embeddings_dict:
        embeddings_dict[omic] = embeddings_dict[omic].numpy()

    save_embeddings(embeddings_dict, labels, os.path.join(config.SAVE_DIR, "embeddings"))
    logger.info("✅ Embeddings and models saved.")
    
    plot_combined_survival_histograms(
        epoch_histograms,
        save_path=os.path.join(config.SAVE_DIR, "combined_survival_histograms.png")
    )


if __name__ == "__main__":
    train()
