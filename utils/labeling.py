import os
import numpy as np
import pandas as pd
from market_config import get_market_config

def save_label_to_dataset_file(dataset_path, action_file, label_column, threshold=0.0001):
    data = pd.read_csv(dataset_path)
    open_prices = data['Open'].values
    close_prices = data['Close'].values
    changes = (close_prices - open_prices) / open_prices

    labels = np.zeros(len(changes), dtype=int)
    labels[changes >= threshold] = 1  # Long
    labels[changes <= -threshold] = 2  # Short
    data[label_column] = labels
    data.to_csv(action_file, index=False)

def save_label_to_q_values_file(market, input_dir, output_dir, action_file, label_column):
    action_df = pd.read_csv(action_file)[["Date", label_column]]
    num_walks = get_market_config(market)["num_walks"]
    phase = ["train", "valid", "test"]

    for i in range(num_walks):
        for p in phase:
            q_values_df = pd.read_csv(f"{input_dir}/{p}/q_values_walk{i}.csv").iloc[2:]

            if 'Date' not in q_values_df.columns:
                q_values_df.reset_index(inplace=True)
                
            q_values_merged_df = q_values_df.merge(action_df, on="Date", how="left")
            q_values_merged_df.to_csv(f"{output_dir}/walk{i}_{p}_labeled.csv", index=False)


if __name__ == "__main__":
    market = "dax"
    model_name = "original"

    label_column = "label"
    dataset_file = f"datasets/{market}Day.csv"
    dataset_with_label_file = f"datasets/{market}Day_labeled.csv"

    input_dir = f"Output/q_values/{market}/{model_name}"
    output_dir = f"Output_moe/data/{market}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save label to dataset file
    save_label_to_dataset_file(dataset_file, dataset_with_label_file, label_column, threshold=0.0001)

    # Label the q_values from the labeled actions
    save_label_to_q_values_file(market, input_dir, output_dir, dataset_with_label_file, label_column)