import pandas as pd
import torch
from torch.utils.data import Dataset

class ExpertDataset(Dataset):
    def __init__(self, input_dir, walk_id, phase):
        context_df = pd.read_csv(f"{input_dir}/{phase}/walk{walk_id}_context.csv")
        qvalues_df = pd.read_csv(f"{input_dir}/{phase}/walk{walk_id}_qvalues.csv")
        reward_df = pd.read_csv(f"{input_dir}/{phase}/walk{walk_id}_reward.csv")
        risk_df = pd.read_csv(f"{input_dir}/{phase}/walk{walk_id}_risk.csv")

        # Load context, date and label
        self.date = context_df['Date']
        self.context = context_df.iloc[:, 1:-1].values.astype('float32')  # f0-f67
        self.label = context_df['Label'].values.astype('int64')

        # Load qvalues (num_samples, 100, 3)
        qvalue_cols = [col for col in qvalues_df.columns if col != 'Date']
        self.qvalue = qvalues_df[qvalue_cols].values.reshape(-1, 100, 3).astype('float32')

        # Load rewards (num_samples, 100, 2)
        reward_cols = [col for col in reward_df.columns if col.endswith("reward") or col.endswith("acc")]
        self.reward = reward_df[reward_cols].values.reshape(-1, 100, 2).astype('float32')

        # Extract risks: (num_samples, 100, 2)
        risk_cols = [col for col in risk_df.columns if col != 'Date']
        self.risk = risk_df[risk_cols].values.reshape(-1, 100, 2).astype('float32')

    def __len__(self):
        return len(self.date)

    def __getitem__(self, idx):
        return (self.date.iloc[idx],
                torch.tensor(self.context[idx], dtype=torch.float),
                torch.tensor(self.qvalue[idx], dtype=torch.float),
                torch.tensor(self.reward[idx], dtype=torch.float),
                torch.tensor(self.risk[idx], dtype=torch.float),
                torch.tensor(self.label[idx], dtype=torch.long))
