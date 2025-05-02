import torch
import os

# General
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
BASE_DIR = './data'
SAVE_DIR = './outputs'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'omicscl_model.pt')

PROCESSED_DIR = "./processed"

# Training hyperparameters
EPOCHS = 1000                # Long training with early stopping
BATCH_SIZE = 32              # Smaller batch → better gradient estimate
LEARNING_RATE = 1e-4         # Cyclical scheduler will override this
WEIGHT_DECAY = 1e-6          # Regularize encoders

SURVIVAL_LOSS_WEIGHT = 10.0

# Contrastive Learning
TEMPERATURE = 0.1            # Lower temp → tighter clusters
EMBEDDING_DIM = 64           # (64 for better c-index, 128 for better purity)
HIDDEN_DIM = 128             # (128 for better c-index, 256 for better purity)

# Cyclical LR (used if ENABLE_CYCLIC_LR is True)
ENABLE_CYCLIC_LR = True
MIN_LR = 1e-5                # Min LR for cyclic
MAX_LR = 1e-3                # Max LR (exploration bound)
STEP_SIZE_UP = 200           # Epochs before peak LR
CYCLE_MOMENTUM = False       # Because Adam has no momentum
CLR_MODE = "triangular2"

# Early Stopping
ENABLE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 20     # Wait 20 epochs of no improvement
EARLY_STOP_DELTA = 1e-4      # Minimum loss decrease or C-index increase

# Visualization
EMBEDDING_PLOT_PATH = os.path.join(SAVE_DIR, 'embeddings_2d.png')
VARIANCE_PLOT_PATH = os.path.join(SAVE_DIR, 'embedding_variance.png')


N_CLUSTERS = 4               # For KMeans clustering in eval(4 for better c-index, 9 for better purity)
