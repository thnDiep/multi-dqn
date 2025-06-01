import os
import random
import numpy as np
import pandas as pd


def label_trading_actions(data, threshold=0.0001):
    # data = data[window_size:]
    y_true = np.ones(len(data), dtype=int)
    last_action = 1

    for i in range(len(data) - 1):
        current_price = data.iloc[i]["Close"]
        next_price = data.iloc[i + 1]["Close"]

        changes = (next_price - current_price) / current_price

        if changes >= threshold or (last_action == 0 and changes >= 0):
            last_action = 1  # BUY
            y_true[i] = 1
        elif changes <= -threshold or (last_action == 2 and changes < 0):
            last_action = 2  # SELL
            y_true[i] = 2
        else:
            last_action = 0  # HOLD
            y_true[i] = 0
    return np.array(y_true)


# data = pd.read_csv("datasets/daxDay.csv") 
# action_labels = label_trading_actions(data)

# data["action_label"] = action_labels
# data.to_csv("datasets/daxDay_actions.csv", index=False)


action_file = "datasets/daxDay_actions.csv"
input_q_value_dir = "Output/q_values/dax/original"
input_action_dir = "Output/ensemble/dax/original"

output_dir = "Output/moe_original_q_value"


# Đọc file action label
action_df = pd.read_csv(action_file)[["Date", "action_label"]]

# Đảm bảo cột Date có định dạng giống các file walk
# Có thể cần chỉnh nếu format khác nhau, ví dụ:
# action_df["Date"] = pd.to_datetime(action_df["Date"]).dt.strftime("%m/%d/%Y")

for i in range(8):
    walk_path = f"{input_q_value_dir}/train/q_values_walk{i}.csv"
    df = pd.read_csv(walk_path)

    if 'Date' not in df.columns:
        df.reset_index(inplace=True)

    # Nối thêm cột Action_Label từ file action
    merged_df = df.merge(action_df, on="Date", how="left")

    # Ghi ra file mới
    merged_df.to_csv(f"{output_dir}/walk{i}_train_labeled.csv", index=False)

    print(f"✅ walk{i} saved with Action_Label.")


for i in range(8):
    walk_path = f"{input_q_value_dir}/valid/q_values_walk{i}.csv"

    df = pd.read_csv(walk_path)

    if 'Date' not in df.columns:
        df.reset_index(inplace=True)

    # Nối thêm cột Action_Label từ file action
    merged_df = df.merge(action_df, on="Date", how="left")

    # Ghi ra file mới
    merged_df.to_csv(f"{output_dir}/walk{i}_valid_labeled.csv", index=False)

    print(f"✅ walk{i} saved with Action_Label.")


for i in range(8):
    walk_path = f"{input_q_value_dir}/test/q_values_walk{i}.csv"
    df = pd.read_csv(walk_path)

    if 'Date' not in df.columns:
        df.reset_index(inplace=True)

    # Nối thêm cột Action_Label từ file action
    merged_df = df.merge(action_df, on="Date", how="left")

    # Ghi ra file mới
    merged_df.to_csv(f"{output_dir}/walk{i}_test_labeled.csv", index=False)

    print(f"✅ walk{i} saved with Action_Label.")