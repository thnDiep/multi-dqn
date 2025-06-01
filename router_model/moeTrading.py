import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from router_model.moeNetwork import MoeNetwork
from evaluation.evaluation import Evaluation
from utils.market_config import get_market_config
from router_model.expertDataset import ExpertDataset

class MoeTrading:
    def __init__(self, market, model_name, model_type, num_epochs=100):
        self.market = market
        self.num_walks = get_market_config(market)['num_walks']
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.model_type = model_type

        # Prepare folder paths
        self.input_dir = f"./Output/labeled/{model_type}/{market}/{model_name}"

        self.ensemble_dir = f"./Output/moe/ensemble/{model_type}/{market}/{model_name}"
        self.model_dir = f"./Output/moe/models/{model_type}/{market}/{model_name}"
        self.result_dir = f"./Output/moe/results/{model_type}/{market}/{model_name}"

        os.makedirs(self.ensemble_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        # Instantiate the agent with parameters received
        self.model = MoeNetwork(model_type=self.model_type)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, dataloader):
        print("Training...")
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for date, expert_data, labels in dataloader:
                final_action, final_distribution, loss = self.model(expert_data, target_action=labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"    Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    def evaluate(self, dataloader, current_walk, phase="test"):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_date = []
        with torch.no_grad():
            for date, expert_data, labels in dataloader:
                predicted, _, _ = self.model(expert_data)
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())
                all_date.extend(date)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"[Evaluation] Accuracy: {accuracy * 100:.2f}%")

        df_result = pd.DataFrame({
            "Date": all_date,
            "ensemble": all_preds,
            "label": all_labels
        })
        df_result.to_csv(f"{self.ensemble_dir}/walk{current_walk}_{phase}.csv", index=False)

        print(f"Accuracy - {phase}: {accuracy*100:.2f}%")
        return correct, total

    def run(self):
        total_correct = 0
        total_samples = 0

        for walk_id in range(self.num_walks):
            print(f"\n Walk {walk_id}")

            # Load dữ liệu
            df_train = pd.read_csv(f"{self.input_dir}/walk{walk_id}_train_labeled.csv")
            df_valid = pd.read_csv(f"{self.input_dir}/walk{walk_id}_valid_labeled.csv")
            df_test  = pd.read_csv(f"{self.input_dir}/walk{walk_id}_test_labeled.csv")

            train_loader = DataLoader(ExpertDataset(df_train, self.model_type), batch_size=32, shuffle=True)
            valid_loader = DataLoader(ExpertDataset(df_valid, self.model_type), batch_size=32, shuffle=False)
            test_loader  = DataLoader(ExpertDataset(df_test, self.model_type),  batch_size=32, shuffle=False)

            self.model = MoeNetwork(model_type=self.model_type)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

            self.train(train_loader)
            self.evaluate(valid_loader, walk_id, phase="valid")

            correct, total = self.evaluate(test_loader, walk_id, phase="test")

            total_correct += correct
            total_samples += total
            torch.save(self.model.state_dict(), f"{self.model_dir}/walk{walk_id}.pth")

        final_accuracy = total_correct / total_samples
        print(f"Overall Test Accuracy across {self.num_walks} walks: {final_accuracy * 100:.2f}%")
        
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

        evaluation.plot_results(ensemble_type="moe")
        print(f"Evaluation completed for {self.market}. Check the Output directory for results.")
