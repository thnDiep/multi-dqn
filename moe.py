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
#     def __init__(self, num_experts=100, num_actions=3, hidden_dim=128, model_type='action', class_weights=None):
#         super().__init__()
#         self.num_experts = num_experts
#         self.num_actions = num_actions
#         self.class_weights = class_weights
#         input_dim = num_experts * num_actions 

#         # self.router = nn.Sequential(
#         #     nn.Linear(input_dim, hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.3),
#         #     nn.Linear(hidden_dim, num_experts),
#         # )

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

#     def forward(self, expert_actions, target_action=None):
#         # expert_actions: (batch_size, num_experts)
#         if self.training:
#             noise = (torch.rand_like(expert_actions.float()) < 0.05).long()
#             expert_actions = (expert_actions + noise) % self.num_actions
#         if model_type == 'action':
#             expert_one_hot = F.one_hot(expert_actions, num_classes=self.num_actions).float()
#             router_input = expert_one_hot.view(expert_one_hot.size(0), -1)  # (batch_size, 100*3)
        
#         router_logits = self.router(router_input)  # (batch_size, num_experts)
#         # expert_weights = F.softmax(router_logits, dim=-1)

#         # expert_one_hot: (batch_size, num_experts, num_actions)
#         weights = router_logits.unsqueeze(-1)  # (batch_size, num_experts, 1)
#         final_distribution = torch.sum(weights * expert_one_hot, dim=1)  # (batch_size, num_actions)
#         # final_distribution = final_distribution / final_distribution.sum(dim=-1, keepdim=True)
        
#         final_action = torch.argmax(final_distribution, dim=-1)

#         loss = None
#         if target_action is not None:
#             # loss = F.cross_entropy(final_distribution, target_action, weight=self.class_weights)
#             loss = F.cross_entropy(final_distribution, target_action, weight=self.class_weights, label_smoothing=0.1)

#             # loss = F.cross_entropy(final_distribution, target_action, weight=self.class_weights)
    
#             # expert_weights = F.softmax(router_logits, dim=-1)
#             # entropy = - (expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
#             # loss = loss - 0.01 * entropy  # 0.01 is a regularization strength, you can tune it
#         return final_action, final_distribution, loss
    

class SoftMoEStock(nn.Module):
    def __init__(self, num_experts=100, num_actions=3, hidden_dim=128, model_type='action', class_weights=None):
        super().__init__()
        self.num_experts = num_experts
        self.num_actions = num_actions
        self.class_weights = class_weights

        self.expert_encoder = nn.Sequential(
            nn.Linear(num_actions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Learnable positional embeddings for experts
        self.positional_embedding = nn.Parameter(torch.randn(1, num_experts, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Router: from context → expert weights
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, expert_actions, target_action=None, lambda_entropy=0.01):
        # if self.training:
        #     noise = (torch.rand_like(expert_actions.float()) < 0.05).long()
        #     expert_actions = (expert_actions + noise) % self.num_actions

        # One-hot encode expert actions: (B, 100, 3)
        expert_one_hot = F.one_hot(expert_actions, num_classes=self.num_actions).float()

        # Encode each one-hot vector
        x = self.expert_encoder(expert_one_hot)  # (B, 100, hidden_dim)
        x = x + self.positional_embedding        # (B, 100, hidden_dim)

        # Pass through Transformer
        x = self.transformer(x)                  # (B, 100, hidden_dim)
        pooled = x.mean(dim=1)                   # (B, hidden_dim)

        # Compute routing weights
        router_logits = self.router_mlp(pooled)  # (B, 100)
        expert_weights = F.softmax(router_logits, dim=-1)  # (B, 100)
        expert_weights_expanded = expert_weights.unsqueeze(-1)  # (B, 100, 1)

        # Combine expert outputs weighted by attention
        final_distribution = torch.sum(expert_weights_expanded * expert_one_hot, dim=1)  # (B, 3)
        final_action = torch.argmax(final_distribution, dim=-1)  # (B,)

        loss = None
        if target_action is not None:
            ce_loss = F.cross_entropy(final_distribution, target_action, weight=self.class_weights, label_smoothing=0.1)
            entropy = - (expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
            loss = ce_loss + lambda_entropy * entropy

        return final_action, final_distribution, loss


def train_moe(model, dataloader, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for date, expert_actions, labels in dataloader:
            final_action, final_distribution, loss = model(expert_actions, target_action=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"    Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
def train_moe(model, dataloader, optimizer, valid_loader=None, num_epochs=50, patience=10):
    best_val_loss = float('inf')
    wait = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for date, expert_actions, labels in dataloader:
            _, _, loss = model(expert_actions, target_action=labels)
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
        for date, expert_actions, labels in dataloader:
            predicted, _, loss = model(expert_actions, target_action=labels)
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



class ExpertActionDataset(Dataset):
    def __init__(self, df):
        self.date = df.iloc[:, 0].values  
        self.expert_actions = df.iloc[:, 1:-1].values.astype(int)   # shape: (N, 100)
        self.labels = df.iloc[:, -1].values.astype(int)              # shape: (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.date[idx], 
            torch.tensor(self.expert_actions[idx], dtype=torch.long),  # (100,)
            torch.tensor(self.labels[idx], dtype=torch.long)           # ()
        )


def run(input_model, df_path, model_type):
    num_walks = 8
    total_correct = 0
    total_samples = 0

    for walk_id in range(num_walks):
        print(f"\n🚶 Walk {walk_id}")

        # Load dữ liệu
        df_train = pd.read_csv(f"{df_path}/walk{walk_id}_train_labeled.csv")
        df_valid = pd.read_csv(f"{df_path}/walk{walk_id}_valid_labeled.csv")
        df_test  = pd.read_csv(f"{df_path}/walk{walk_id}_test_labeled.csv")

        train_loader = DataLoader(ExpertActionDataset(df_train), batch_size=32, shuffle=True)
        valid_loader = DataLoader(ExpertActionDataset(df_valid), batch_size=32, shuffle=False)
        test_loader  = DataLoader(ExpertActionDataset(df_test),  batch_size=32, shuffle=False)

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
        model = SoftMoEStock(num_experts=100, num_actions=3, hidden_dim=128, model_type=model_type, class_weights=class_weights)
        # optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        print("[Training]")
        # train_moe(model, train_loader, optimizer, num_epochs=50)
        train_moe(model, train_loader, optimizer, valid_loader=valid_loader, num_epochs=50)
        print("[Validation]")
        val_loss, val_acc, _, _ = evaluate_moe(model, valid_loader, save_path=f"{df_path}/test_action/walk{walk_id}_valid.csv")
        print(f"    Validation Loss: {val_loss:.4f}")
        print(f"    Validation Accuracy: {val_acc*100:.2f}%")

        print("[Testing]")
        test_loss, test_acc, correct, total = evaluate_moe(model, test_loader, save_path=f"{df_path}/test_action/walk{walk_id}_test.csv")
        print(f"    Test Loss: {test_loss:.4f}%")
        print(f"    Test Accuracy: {test_acc*100:.2f}%")

        total_correct += correct
        total_samples += total

    final_accuracy = total_correct / total_samples
    print(f"Overall Test Accuracy across 8 walks: {final_accuracy * 100:.2f}%")
    
    # Save the model
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
                    # r = (close_price - open_price) / open_price
                    # rew += r
                    # pos += 1 if r > 0 else 0
                    # neg += 0 if r > 0 else 1
                    # doll += (close_price - open_price) * 50
                    # cov += 1
                    rew+= (close_price-open_price)/open_price
                    pos+= 1 if rew > 0 else 0
                    neg+= 0 if rew > 0 else 1
                    doll+=(close_price-open_price)*50
                    cov+=1
                elif action == 2:  # Short
                    # r = -(close_price - open_price) / open_price
                    # rew += r
                    # pos += 1 if r > 0 else 0
                    # neg += 0 if r > 0 else 1
                    # doll += -(close_price - open_price) * 50
                    # cov += 1
                    rew+=-(close_price-open_price)/open_price
                    neg+= 0 if -rew > 0 else 1
                    pos+= 1 if -rew > 0 else 0
                    cov+=1
                    doll+=-(close_price-open_price)*50
        

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

df_path = "Output/moe_original"
input_model = "original"
market_file = "datasets/daxDay.csv"

# action || q_value
model_type = "action"

run(input_model, df_path, model_type)
evaluation(df_path, market_file)