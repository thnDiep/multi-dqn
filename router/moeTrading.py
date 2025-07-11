import os
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from router.moeNetwork import MoERouter
from evaluation.evaluation import Evaluation
from utils.market_config import get_market_config
from router.expertDataset import ExpertDataset

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MoeTrading:
    def __init__(self, market, model_name, num_epochs=100, moe_model_type="flat", lr=1e-3, weight_decay=1e-5, label_smoothing=0.01, lambda_entropy=0.1):
        self.optimizer = None
        self.model = None
        self.market = market
        self.num_walks = get_market_config(market)['num_walks']
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.moe_model_type = moe_model_type
        self.patience = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.lambda_entropy = lambda_entropy

        # Prepare folder paths
        self.input_dir = f"./Output_moe/data/{market}/{model_name}"
        self.ensemble_dir = f"./Output_moe/ensemble/{market}/{model_name}_{moe_model_type}"
        self.model_dir = f"./Output_moe/models/{market}/{model_name}_{moe_model_type}"
        self.result_dir = f"./Output_moe/results/{market}/"
    
        os.makedirs(self.ensemble_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)


    def train(self, walk_id, data_loader, verbose=True):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            total_samples = 0

            for date, context, qvalue, reward, risk, label in data_loader:
                context = context.to(self.device)
                qvalue = qvalue.to(self.device)
                reward = reward.to(self.device)
                risk = risk.to(self.device)
                label = label.to(self.device)
                
                self.optimizer.zero_grad()
                action_logits, gate_weights, expert_logits = self.model(context, qvalue, reward, risk)
                loss = self.loss_function(action_logits, label, expert_logits, gate_weights)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * label.size(0)
                total_samples += label.size(0)
                
            train_loss = total_loss / total_samples
          
            if verbose:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")

        torch.save(self.model.state_dict(), f"{self.model_dir}/walk{walk_id}.pth")


    def evaluate(self, walk_id, data_loader, phase="test", verbose=True):
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        total_correct = 0

        all_date = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for date, context, qvalue, reward, risk, labels in data_loader:
                context = context.to(self.device)
                qvalue = qvalue.to(self.device)
                reward = reward.to(self.device)
                risk = risk.to(self.device)
                labels = labels.to(self.device)
            
                action_logits, gate_weights, expert_logits = self.model(context, qvalue, reward, risk)
                predicted = self.model.get_action(action_logits)
                loss = self.loss_function(action_logits, labels, expert_logits, gate_weights)

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
    
    def loss_function(self, action_logits, labels, expert_logits, gate_weights, alpha_kl=0.01, alpha_entropy=0.1):
        # Task loss - loss chính cho việc dự đoán action
        task_loss = F.cross_entropy(action_logits, labels)

        # KL divergence - khuyến khích phân phối expert đồng đều
        uniform = torch.full_like(expert_logits, 1.0 / expert_logits.size(-1))
        kl_loss = F.kl_div(F.log_softmax(expert_logits, dim=-1), uniform, reduction='batchmean')

        # Entropy regularization - khuyến khích đa dạng trong việc chọn expert
        entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1).mean()

        # Total loss
        total_loss = task_loss + alpha_kl * kl_loss + alpha_entropy * entropy
        return total_loss
    
    def run(self):
        total_correct = 0
        total_samples = 0

        for walk_id in range(self.num_walks):
            print(f"\nWalk {walk_id}")

            train_loader = DataLoader(ExpertDataset(self.input_dir, walk_id, phase="valid"), batch_size=32)
            test_loader  = DataLoader(ExpertDataset(self.input_dir, walk_id, phase="test"),  batch_size=32)

            self.model = MoERouter(num_experts=100).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            self.train(walk_id, train_loader)
            _, correct, total = self.evaluate(walk_id, test_loader, phase="test")

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

        evaluation.plot_results(ensemble_type="moe", moe_model_type=self.moe_model_type)
        print(f"Evaluation completed for {self.market}. Check the Output directory for results.")
