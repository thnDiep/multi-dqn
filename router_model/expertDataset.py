import numpy as np
import torch
from torch.utils.data import Dataset

class ExpertDataset(Dataset):
    def __init__(self, df, model_type='action'):
        self.date = df["Date"].values
        self.model_type = model_type
        
        if model_type == 'action':
            self.expert_data = df.iloc[:, 1:-1].values.astype(int)  # shape: (N, 100)
        else:  # q_value
            qvalue_cols = [col for col in df.columns if col.startswith("iteration")]
            self.expert_data = df[qvalue_cols].values.reshape(-1, 100, 3).astype(np.float32)
            
        self.labels = df.iloc[:, -1].values.astype(int) # shape: (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.date[idx],
            torch.tensor(self.expert_data[idx], dtype=torch.float if self.model_type == 'q_values' else torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
