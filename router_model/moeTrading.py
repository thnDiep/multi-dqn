import os
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from router_model.moeNetwork import MoeFlatNetwork, Moe2DNetwork
from evaluation.evaluation import Evaluation
from utils.market_config import get_market_config
from router_model.expertDataset import ExpertDataset

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MoeTrading:
    def __init__(self, market, model_name, model_type, num_epochs=100, moe_model_type="flat"):
        self.market = market
        self.num_walks = get_market_config(market)['num_walks']
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.model_type = model_type
        self.moe_model_type = moe_model_type
        self.patience = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare folder paths
        self.input_dir = f"./Output/labeled/{model_type}/{market}/{model_name}"

        self.ensemble_dir = f"./Output/moe/ensemble/{model_type}/{market}/{model_name}"
        self.model_dir = f"./Output/moe/models/{model_type}/{market}/{model_name}"
        self.result_dir = f"./Output/moe/results/{model_type}/{market}/{model_name}"

        os.makedirs(self.ensemble_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        # Instantiate the agent with parameters received
        if self.moe_model_type == "flat":
            self.model = MoeFlatNetwork(model_type=self.model_type).to(self.device)
        elif self.moe_model_type == "2d":
            self.model = Moe2DNetwork(model_type=self.model_type).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def train(self, walk_id, train_loader, valid_loader, class_weights, verbose=True):
        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            total_samples = 0

            for date, expert_data, labels in train_loader:
                expert_data = expert_data.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs, expert_weights = self.model(expert_data)
                loss = self.loss_function(outputs, expert_weights, labels, class_weights)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                
            train_loss = total_loss / total_samples
            val_loss, _, _ = self.evaluate(walk_id, valid_loader, class_weights, phase="valid", verbose=False)

            if verbose:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}") 
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            if best_state:
                self.model.load_state_dict(best_state)
                torch.save(best_state, f"{self.model_dir}/walk{walk_id}.pth")

    def evaluate(self, walk_id, data_loader, class_weights, phase="test", verbose=True):
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        total_correct = 0

        all_date = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for date, expert_data, labels in data_loader:
                expert_data = expert_data.to(self.device)
                labels = labels.to(self.device)

                outputs, expert_weights = self.model(expert_data)
                predicted = self.model.get_action(outputs)
                loss = self.loss_function(outputs, expert_weights, labels, class_weights)

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                
                all_date.extend(date)
                all_labels.extend(labels.tolist())
                all_preds.extend(predicted.tolist())

        avg_loss = total_loss / total_samples
        if verbose:
            accuracy = total_correct / total_samples
            df_result = pd.DataFrame({
                "Date": all_date,
                "ensemble": all_preds,
                "label": all_labels
            })
            df_result.to_csv(f"{self.ensemble_dir}/walk{walk_id}_{phase}.csv", index=False)

            print(f"[Evaluation] Loss - {phase}: {avg_loss:.4f}")
            print(f"[Evaluation] Accuracy - {phase}: {accuracy*100:.2f}%")
        return avg_loss, total_correct, total_samples
    
    def loss_function(self, outputs, expert_weights, labels, weight=None, label_smoothing=0.1, lambda_entropy=0.01):
        ce_loss = F.cross_entropy(outputs, labels, weight=weight, label_smoothing=label_smoothing)
        entropy = - (expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
        return ce_loss + lambda_entropy * entropy
    
    def run(self):
        total_correct = 0
        total_samples = 0

        for walk_id in range(self.num_walks):
            print(f"\nWalk {walk_id}")

            # Load dữ liệu
            df_train = pd.read_csv(f"{self.input_dir}/walk{walk_id}_train_labeled.csv")
            df_valid = pd.read_csv(f"{self.input_dir}/walk{walk_id}_valid_labeled.csv")
            df_test  = pd.read_csv(f"{self.input_dir}/walk{walk_id}_test_labeled.csv")

            print(df_train.head())  # Kiểm tra 5 dòng đầu tiên của DataFrame
            print(df_train.shape)   # Kiểm tra kích thước của DataFrame
            print(df_train.columns)  # Kiểm tra danh sách các cột
            train_loader = DataLoader(ExpertDataset(df_train, self.model_type), batch_size=32, shuffle=True)
            valid_loader = DataLoader(ExpertDataset(df_valid, self.model_type), batch_size=32, shuffle=False)
            test_loader  = DataLoader(ExpertDataset(df_test, self.model_type),  batch_size=32, shuffle=False)

            class_weights = self.calculate_class_weights(df_train)

            if self.moe_model_type == "flat":
                self.model = MoeFlatNetwork(model_type=self.model_type).to(self.device)
            elif self.moe_model_type == "2d":
                self.model = Moe2DNetwork(model_type=self.model_type).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

            self.train(walk_id, train_loader, valid_loader, class_weights)
            self.evaluate(walk_id, valid_loader, class_weights, phase="valid")

            _, correct, total = self.evaluate(walk_id, test_loader, class_weights, phase="test")

            total_correct += correct
            total_samples += total

        final_accuracy = total_correct / total_samples
        print(f"Overall Test Accuracy across {self.num_walks} walks: {final_accuracy * 100:.2f}%")
    
    def calculate_class_weights(self, df):
        label_counts = Counter(df.iloc[:, -1])
        total = sum(label_counts.values())
        num_classes = 3

        class_weights = [0.0] * num_classes
        for i in range(num_classes):
            freq = label_counts.get(i, 0)
            class_weights[i] = total / (freq + 1e-6)  # Tránh chia cho 0
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        return class_weights

    # Function to end the Agent
    def end(self):
        """End the training process and create an evaluation report"""

        evaluation = Evaluation(
            model_name=self.model_name,
            market=self.market,
            input_dir=self.ensemble_dir,
            result_dir=self.result_dir,
            final_decision_dir=None,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )

        evaluation.plot_results(ensemble_type="moe", model_type=self.model_type, moe_model_type=self.moe_model_type)
        print(f"Evaluation completed for {self.market}. Check the Output directory for results.")
