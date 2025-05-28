import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

class SoftMoEStock(nn.Module):
    def __init__(self, num_experts=100, num_actions=3, hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.num_actions = num_actions

        input_dim = num_experts * num_actions  # Do dùng trực tiếp Q-values

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_qvalues, target_action=None):
        # expert_qvalues: (batch_size, num_experts, num_actions)
        batch_size = expert_qvalues.size(0)

        router_input = expert_qvalues.view(batch_size, -1)  # (batch_size, 100*3)
        expert_weights = self.router(router_input)          # (batch_size, num_experts)
        
        weights = expert_weights.unsqueeze(-1)              # (batch_size, num_experts, 1)
        final_distribution = torch.sum(weights * expert_qvalues, dim=1)  # (batch_size, num_actions)
        eps = 1e-9
        final_distribution = final_distribution / (final_distribution.sum(dim=-1, keepdim=True) + eps)
        final_action = torch.argmax(final_distribution, dim=-1)

        loss = None
        if target_action is not None:
            loss = F.cross_entropy(final_distribution, target_action)

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
    


def evaluate_moe(model, dataloader, save_path=None):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_date = []
    with torch.no_grad():
        for date, expert_actions, labels in dataloader:
            predicted, _, _ = model(expert_actions)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
            all_date.extend(date)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"[Evaluation] Accuracy: {accuracy * 100:.2f}%")

    if save_path:
        df_result = pd.DataFrame({
            "Date": all_date,
            "final_action": all_preds,
            "label": all_labels
        })
        df_result.to_csv(save_path, index=False)
        print(f"Saved predictions to: {save_path}")

    return accuracy, correct, total



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


def run():
    num_walks = 8
    total_correct = 0
    total_samples = 0

    for walk_id in range(num_walks):
        print(f"\n🚶 Walk {walk_id}")

        # Load dữ liệu
        df_train = pd.read_csv(f"Output/moe_local_feature_atn_q_value/walk{walk_id}_train_labeled.csv")
        df_valid = pd.read_csv(f"Output/moe_local_feature_atn_q_value/walk{walk_id}_valid_labeled.csv")
        df_test  = pd.read_csv(f"Output/moe_local_feature_atn_q_value/walk{walk_id}_test_labeled.csv")

        print(df_train.shape, df_valid.shape, df_test.shape)
        print(df_train.head())
        train_loader = DataLoader(ExpertQValueDataset(df_train), batch_size=32, shuffle=True)
        valid_loader = DataLoader(ExpertQValueDataset(df_valid), batch_size=32, shuffle=False)
        test_loader  = DataLoader(ExpertQValueDataset(df_test),  batch_size=32, shuffle=False)

        # Khởi tạo lại model cho mỗi walk
        model = SoftMoEStock(num_experts=100, num_actions=3, hidden_dim=128)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        print("Training:")
        train_moe(model, train_loader, optimizer, num_epochs=100)

        print("Validation:")
        val_acc, _, _ = evaluate_moe(model, valid_loader, save_path=f"Output/moe_local_feature_atn_q_value/test_action/walk{walk_id}_valid.csv")
        print(f"    Validation Accuracy: {val_acc*100:.2f}%")

        print("Testing:")
        test_acc, correct, total = evaluate_moe(model, test_loader, save_path=f"Output/moe_local_feature_atn_q_value/test_action/walk{walk_id}_test.csv")
        print(f"    Test Accuracy: {test_acc*100:.2f}%")

        total_correct += correct
        total_samples += total

    final_accuracy = total_correct / total_samples
    print(f"Overall Test Accuracy across 8 walks: {final_accuracy * 100:.2f}%")



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

def evaluation():
    plot_ensemble_results(
        num_walks=8,
        result_file="./Output/moe_local_feature_atn_q_value/actions_result.pdf",
        action_folder="./Output/moe_local_feature_atn_q_value/test_action",
        market_file="./datasets/daxDay.csv",
    )


run()
evaluation()