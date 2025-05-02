import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Computes NT-Xent loss for one batch of z1 and z2 embeddings.
    z1: [N, D], z2: [N, D]
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2N, 2N]

    # Mask self-similarity
    self_mask = torch.eye(2 * N, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(self_mask, -9e15)

    # Positive pairs (i, i+N) and (i+N, i)
    pos = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)

    logits = sim / temperature
    labels = pos

    loss = F.cross_entropy(logits, labels)
    return loss

def survival_contrastive_loss(embeddings, times, events, margin=1.0, scale=1.0):
    """
    Unsupervised survival-aware contrastive loss.
    - embeddings: Tensor of shape (N, D)
    - times: Tensor of shape (N,)
    - events: Tensor of shape (N,)
    - margin: Time difference threshold to define dissimilarity
    - scale: Distance scaling factor
    """
    N = embeddings.size(0)
    dists = torch.cdist(embeddings, embeddings, p=2)  # Pairwise Euclidean distance
    time_diff = torch.abs(times.unsqueeze(1) - times.unsqueeze(0))  # (N x N)

    # Similarity matrix: both died AND time difference < margin
    both_died = (events == 1).float().unsqueeze(1) * (events == 1).float().unsqueeze(0)
    sim_mask = (time_diff < margin) * both_died  # (N x N)

    # Pull similar pairs
    pull = scale * (dists * sim_mask).pow(2).mean()

    # Push dissimilar pairs
    dissim_mask = (time_diff >= margin).float()
    push = scale * F.relu(margin - dists) * dissim_mask
    push = push.pow(2).mean()

    return pull + push