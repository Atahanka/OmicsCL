import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MultiOmicsDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        self.gene = torch.tensor(data["gene"], dtype=torch.float32)
        self.meth = torch.tensor(data["meth"], dtype=torch.float32)
        self.mirna = torch.tensor(data["mirna"], dtype=torch.float32)
        self.time = torch.tensor(data["time"], dtype=torch.float32)
        self.event = torch.tensor(data["event"], dtype=torch.float32)
        self.subtype = data["subtype"] if "subtype" in data.files else None

    def __len__(self):
        return len(self.time)

    def __getitem__(self, idx):
        return {
            "gene": self.gene[idx],
            "meth": self.meth[idx],
            "mirna": self.mirna[idx],
            "time": self.time[idx],
            "event": self.event[idx],
            "subtype": self.subtype[idx]
        }

def get_dataloaders(data_dir="processed", batch_size=32, shuffle=True):
    train_set = MultiOmicsDataset(os.path.join(data_dir, "train.npz"))
    val_set = MultiOmicsDataset(os.path.join(data_dir, "val.npz"))
    test_set = MultiOmicsDataset(os.path.join(data_dir, "test.npz"))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
