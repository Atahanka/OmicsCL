import os
import random
import numpy as np
import torch
import logging
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score


def set_seed(seed=42):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dir(path):
    """
    Create a directory if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(model, path):
    """
    Save a PyTorch model to the specified path.
    """
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cuda'):
    """
    Load a PyTorch model from a file.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_embeddings(embeddings_dict, labels, save_dir):
    """
    Save embeddings and labels as .npy files.
    Args:
        embeddings_dict: dict of {omics_name: numpy array of shape (N, D)}
        labels: array of shape (N,)
        save_dir: where to save the files
    """
    ensure_dir(save_dir)
    for view, emb in embeddings_dict.items():
        np.save(os.path.join(save_dir, f"{view}_embeddings.npy"), emb)
    np.save(os.path.join(save_dir, "labels.npy"), labels)


def compute_clustering_metrics(y_true, y_pred):
    """
    Compute common clustering evaluation metrics.
    Returns:
        accuracy (if classes are aligned), ARI, NMI
    """
    acc = accuracy_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return acc, ari, nmi


def setup_logger(log_path=None):
    """
    Set up logging to both console and optionally to a file.
    Returns:
        logger instance
    """
    logger = logging.getLogger("OmicsCL")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Prevent adding duplicate handlers if this function is called multiple times
    if not logger.handlers:
        # Console logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File logging
        if log_path:
            ensure_dir(os.path.dirname(log_path))
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
