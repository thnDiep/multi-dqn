import os
import numpy as np
import pandas as pd
from market_config import get_market_config

def label_trading_actions(data, threshold=0.0001):
    open_prices = data['Open'].values
    close_prices = data['Close'].values
    changes = (close_prices - open_prices) / open_prices

    labels = np.zeros(len(changes), dtype=int)
    labels[changes >= threshold] = 1  # Long
    labels[changes <= -threshold] = 2  # Short
    return labels


data = pd.read_csv("datasets/daxDay.csv") 
action_labels = label_trading_actions(data)

data["action_label"] = action_labels
data.to_csv("datasets/daxDay_actions.csv", index=False)


market = "dax"
model_name = "original"
num_walks = get_market_config(market)["num_walks"]
phase = ["train", "valid", "test"]

action_file = f"datasets/{market}Day_actions.csv"
action_df = pd.read_csv(action_file)[["Date", "action_label"]]

input_q_value_dir = f"Output/q_values/{market}/{model_name}"
input_action_dir = f"Output/ensemble/{market}/{model_name}"

q_values_output_dir = f"Output/labeled/q_values/{market}/{model_name}"
action_output_dir = f"Output/labeled/action/{market}/{model_name}"

os.makedirs(q_values_output_dir, exist_ok=True)
os.makedirs(action_output_dir, exist_ok=True)

for i in range(num_walks):
    for p in phase:
        q_values_df = pd.read_csv(f"{input_q_value_dir}/{p}/q_values_walk{i}.csv").iloc[2:]
        action_values_df = pd.read_csv(f"{input_action_dir}/walk{i}ensemble_{p}.csv").iloc[2:]

        if 'Date' not in q_values_df.columns:
            q_values_df.reset_index(inplace=True)

        if 'Date' not in action_values_df.columns:
            action_values_df.reset_index(inplace=True)

        # Nối thêm cột Action_Label từ file action
        q_values_merged_df = q_values_df.merge(action_df, on="Date", how="left")
        action_merged_df = action_values_df.merge(action_df, on="Date", how="left")

        # Ghi ra file mới
        q_values_merged_df.to_csv(f"{q_values_output_dir}/walk{i}_{p}_labeled.csv", index=False)
        action_merged_df.to_csv(f"{action_output_dir}/walk{i}_{p}_labeled.csv", index=False)