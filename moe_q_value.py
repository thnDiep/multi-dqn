import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import random
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# class SoftMoEStock(nn.Module):
#     def __init__(self, num_experts=100, num_actions=3, hidden_dim=128, class_weights=None):
#         super().__init__()
#         self.num_experts = num_experts
#         self.num_actions = num_actions
#         self.class_weights = class_weights
#         input_dim = num_experts * num_actions 

#         self.router = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             # nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             # nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_experts),
#         )


#     def forward(self, expert_qvalues, target_action=None):
#         # expert_qvalues: (batch_size, num_experts, num_actions)
#         batch_size = expert_qvalues.size(0)

#         router_input = expert_qvalues.view(batch_size, -1)  # (batch_size, 100*3)
#         router_logits = self.router(router_input)          # (batch_size, num_experts)
#         expert_weights = F.softmax(router_logits, dim=-1)

#         weights = expert_weights.unsqueeze(-1) 
#         final_distribution = torch.sum(weights * expert_qvalues, dim=1)  # (batch_size, num_actions)
#         # final_distribution = F.softmax(final_distribution, dim=-1)  # Normalize to get probabilities
#         final_action = torch.argmax(final_distribution, dim=-1)
#         loss = None
#         if target_action is not None:
#             # target_dist = F.one_hot(target_action, num_classes=self.num_actions).float()
#             # loss = F.cross_entropy(final_distribution, target_action)
#             loss = F.cross_entropy(final_distribution, target_action, weight=self.class_weights, label_smoothing=0.1)

#             # log_mixture = torch.log(final_distribution + 1e-9)
#             # loss = F.kl_div(log_mixture, target_dist, reduction="batchmean")
#             entropy = - (expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
#             λ = 0.001 
#             loss = loss + λ * entropy
#         return final_action, final_distribution, loss


class SoftMoEStock(nn.Module):
    def __init__(self, num_experts=100, num_actions=3, hidden_dim=128, class_weights=None):
        super().__init__()
        self.num_experts = num_experts
        self.num_actions = num_actions
        self.class_weights = class_weights

        # Encode Q-values for each expert: [3] → [hidden_dim]
        self.expert_encoder = nn.Sequential(
            nn.Linear(num_actions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Learnable positional encoding
        self.positional_embedding = nn.Parameter(torch.randn(1, num_experts, hidden_dim))

        # Transformer: use 1 layer for stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # MLP router: from pooled context → expert weight logits
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, expert_qvalues, target_action=None, lambda_entropy=0.1):
        # expert_qvalues: (B, 100, 3)
        B = expert_qvalues.size(0)

        x = self.expert_encoder(expert_qvalues)               # (B, 100, hidden_dim)
        x = x + self.positional_embedding                     # (B, 100, hidden_dim)
        x = self.transformer(x)                               # (B, 100, hidden_dim)

        pooled = x.mean(dim=1)                                # (B, hidden_dim)
        router_logits = self.router_mlp(pooled)               # (B, 100)
        expert_weights = F.softmax(router_logits, dim=-1)     # (B, 100)
        expert_weights_expanded = expert_weights.unsqueeze(-1)  # (B, 100, 1)

        # Weighted Q-values
        final_distribution = torch.sum(expert_weights_expanded * expert_qvalues, dim=1)  # (B, 3)
        final_action = torch.argmax(final_distribution, dim=-1)  # (B,)

        loss = None
        if target_action is not None:
            ce_loss = F.cross_entropy(final_distribution, target_action, weight=self.class_weights, label_smoothing=0.1)
            entropy = - (expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
            loss = ce_loss + lambda_entropy * entropy

        return final_action, final_distribution, loss





def train_moe(model, dataloader, optimizer, valid_loader=None, num_epochs=50, patience=10):
    best_val_loss = float('inf')
    wait = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for date, expert_qvale, labels in dataloader:
            final_action, final_distribution, loss = model(expert_qvale, target_action=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"    Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
        if valid_loader:
            val_loss, _, _, _ = evaluate_moe(model, valid_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break



def evaluate_moe(model, dataloader, save_path=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    all_date = []
    with torch.no_grad():
        for date, expert_qvale, labels in dataloader:
            predicted, _, loss = model(expert_qvale, target_action=labels)
            total_loss += loss.item()
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
            all_date.extend(date)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    # print(f"[Evaluation] Accuracy: {accuracy * 100:.2f}%")

    if save_path:
        df_result = pd.DataFrame({
            "Date": all_date,
            "final_action": all_preds,
            "label": all_labels
        })
        df_result.to_csv(save_path, index=False)
        print(f"Saved predictions to: {save_path}")

    return avg_loss, accuracy, correct, total



class ExpertQValueDataset(Dataset):
    def __init__(self, df, label_series=None):
        self.date = df["Date"].values
        qvalue_cols = [col for col in df.columns if col.startswith("iteration")]
        self.qvalues = df[qvalue_cols].values.reshape(-1, 100, 3).astype(np.float32)
        self.labels = df.iloc[:, -1].values.astype(int)   

    def __len__(self):
        return len(self.qvalues)

    def __getitem__(self, idx):
        return (
            self.date[idx],
            torch.tensor(self.qvalues[idx], dtype=torch.float),  # shape: (100, 3)
            torch.tensor(self.labels[idx], dtype=torch.long)     # shape: ()
        )


def run(input_model, df_path):
    num_walks = 8
    total_correct = 0
    total_samples = 0

    for walk_id in range(num_walks):
        print(f"\n🚶 Walk {walk_id}")

        # Load dữ liệu
        df_train = pd.read_csv(f"{df_path}/walk{walk_id}_train_labeled.csv")
        df_valid = pd.read_csv(f"{df_path}/walk{walk_id}_valid_labeled.csv")
        df_test  = pd.read_csv(f"{df_path}/walk{walk_id}_test_labeled.csv")

        train_loader = DataLoader(ExpertQValueDataset(df_train), batch_size=32, shuffle=True)
        valid_loader = DataLoader(ExpertQValueDataset(df_valid), batch_size=32, shuffle=False)
        test_loader  = DataLoader(ExpertQValueDataset(df_test),  batch_size=32, shuffle=False)

        label_counts = Counter(df_train.iloc[:, -1])
        label_counts = Counter(df_train.iloc[:, -1])  # Cột label là cột cuối cùng
        total = sum(label_counts.values())
        num_classes = 3

        class_weights = [0.0] * num_classes
        for i in range(num_classes):
            freq = label_counts.get(i, 0)
            class_weights[i] = total / (freq + 1e-6)  # Tránh chia cho 0
        class_weights = torch.tensor(class_weights, dtype=torch.float)


        # Khởi tạo lại model cho mỗi walk
        model = SoftMoEStock(num_experts=100, num_actions=3, hidden_dim=128, class_weights=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        print("[Training]")
        train_moe(model, train_loader, optimizer, valid_loader=valid_loader, num_epochs=50)

        print("[Validation]")
        val_loss, val_acc, _, _ = evaluate_moe(model, valid_loader, save_path=f"{df_path}/test_action/walk{walk_id}_valid.csv")
        print(f"    Validation Loss: {val_loss:.4f}%")
        print(f"    Validation Accuracy: {val_acc*100:.2f}%")

        print("[Testing]")
        test_loss, test_acc, correct, total = evaluate_moe(model, test_loader, save_path=f"{df_path}/test_action/walk{walk_id}_test.csv")
        print(f"    Test Loss: {test_loss:.4f}%")
        print(f"    Test Accuracy: {test_acc*100:.2f}%")

        total_correct += correct
        total_samples += total

    final_accuracy = total_correct / total_samples
    print(f"Overall Test Accuracy across 8 walks: {final_accuracy * 100:.2f}%")
    
    torch.save(model.state_dict(), f"Output/moe_model/{input_model}.pth")
    print(f"Model saved at Output/moe_{input_model}.pth")


def evaluate_from_final_prediction(action_folder, market_file, num_walks, type='test'):
    
    values = []
    columns = ["Iteration", "Reward%", "#Wins", "#Losses", "Dollars", "Coverage", "Accuracy"]

    market_data = pd.read_csv(market_file, index_col='Date')
    
    total_rew = total_pos = total_neg = total_doll = total_cov = total_num = 0

    for j in range(num_walks):
        df = pd.read_csv(f"{action_folder}/walk{j}_{type}.csv", index_col='Date')
        num = rew = pos = neg = doll = cov = 0

        for date, row in df.iterrows():
            num += 1
            if date in market_data.index:
                open_price = market_data.at[date, 'Open']
                close_price = market_data.at[date, 'Close']
                action = row['final_action']

                if action == 1:  # Long
                    r = (close_price - open_price) / open_price
                    rew += r
                    pos += 1 if r > 0 else 0
                    neg += 0 if r > 0 else 1
                    doll += (close_price - open_price) * 50
                    cov += 1
                    # rew+= (close_price-open_price)/open_price
                    # pos+= 1 if rew > 0 else 0
                    # neg+= 0 if rew > 0 else 1
                    # doll+=(close_price-open_price)*50
                    # cov+=1
                elif action == 2:  # Short
                    r = -(close_price - open_price) / open_price
                    rew += r
                    pos += 1 if r > 0 else 0
                    neg += 0 if r > 0 else 1
                    doll += -(close_price - open_price) * 50
                    cov += 1
                    # rew+=-(close_price-open_price)/open_price
                    # neg+= 0 if -rew > 0 else 1
                    # pos+= 1 if -rew > 0 else 0
                    # cov+=1
                    # doll+=-(close_price-open_price)*50
        

        acc = pos / cov if cov > 0 else 0
        cov_rate = cov / num if num > 0 else 0
        values.append([j, round(rew, 2), pos, neg, round(doll, 1), round(cov_rate, 2), round(acc, 2)])

        total_rew += rew
        total_pos += pos
        total_neg += neg
        total_doll += doll
        total_cov += cov
        total_num += num

    values.append([
        "sum",
        round(total_rew, 2),
        total_pos,
        total_neg,
        round(total_doll, 1),
        round(total_cov / total_num, 2),
        round(total_pos / total_cov, 2) if total_cov > 0 else 0
    ])
    return values, columns


def plot_ensemble_results(num_walks, market_file, action_folder, result_file):
    """Plot ensemble results tables with different thresholds"""
    pdf = PdfPages(result_file)
    
    # thresholds = [0, 0.9, 0.8, 0.7, 0.6]
    # titles = ["FULL ENSEMBLE", "90% ENSEMBLE", "80% ENSEMBLE", "70% ENSEMBLE", "60% ENSEMBLE"]
    
    # for threshold, title in zip(thresholds, titles):
    plt.figure(figsize=(10, 5))
    
    # Validation results
    plt.subplot(1, 2, 1)
    plt.axis('off')
    val, col = evaluate_from_final_prediction(
        action_folder,
        market_file,
        num_walks=8,
        type="valid"
    )
    t = plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(6)
    plt.title("Valid")
    
    # Test results
    plt.subplot(1, 2, 2)
    plt.axis('off')
    val, col = evaluate_from_final_prediction(
        action_folder,
        market_file,
        num_walks=8,
        type="test"
    )

    t = plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(6)
    plt.title("Test")
    
    pdf.savefig()
    
    pdf.close()

def evaluation(df_path, market_file):
    plot_ensemble_results(
        num_walks=8,
        result_file=f"{df_path}/actions_result.pdf",
        action_folder=f"{df_path}/test_action",
        market_file=market_file,
    )


df_path = "Output/moe_original_q_value"
input_model = "original"
market_file = "datasets/daxDay.csv"


run(input_model, df_path)
evaluation(df_path, market_file)