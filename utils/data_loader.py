import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np

from utils.market_config import get_market_config
from utils.action_enum import Action
from environments.mergedDataStructure import MergedDataStructure

class DataLoader:
    def __init__(self, market, model_name, num_experts, rolling_windows=20, observation_window=40):
        self.market = market
        self.model_name = model_name
        self.market_config = get_market_config(market)
        self.num_walks = self.market_config['num_walks']
        self.num_experts = num_experts
        self.rolling_windows = rolling_windows
        self.observation_window = observation_window

        self.ensemble_dir = f"./Output/ensemble/{market}/{model_name}"
        self.q_values_dir = f"./Output/q_values/{market}/{model_name}"
        self.output_dir = f"./Output_moe/data/{market}/{model_name}"

        self.price_df = pd.read_csv(f"./datasets/{market}Day.csv")
        self.price_df['Date'] = pd.to_datetime(self.price_df['Date'])

        # Converts each column to a list
        self.hourly_df = pd.read_csv(f"./datasets/{market}Hour.csv")
        self.history = self.hourly_df.to_dict(orient='records')
        self.history_index = {
            pd.to_datetime(row['Date']): idx for idx, row in enumerate(self.history)
        }

        # weekly & daily data structures
        self.dayData = MergedDataStructure(delta=20, filename=f"./datasets/{market}Day.csv")
        self.weekData = MergedDataStructure(delta=8, filename=f"./datasets/{market}Week.csv")

    def _find_history_index(self, date):
        try:
            return self.history_index[date]
        except KeyError:
            raise ValueError(f"Date {date} not found in hourly history")

    def get_context_features(self, dates):
        """
        Context features (40 hourly + 20 daily + 8 weekly)
        """
        rows = []
        for date in dates:
            current_date = pd.to_datetime(date).date()
            prev_indices = [
                (d, idx) for d, idx in self.history_index.items()
                if d.date() < current_date
            ]

            if not prev_indices:
                raise ValueError(f"No history before {date}")

            prev_idx = max(prev_indices, key=lambda x: x[0])[1]
            hourly = self.history[prev_idx - self.observation_window + 1 : prev_idx + 1]

            date_str = current_date.strftime("%m/%d/%Y")
            daily = self.dayData.get(date_str)
            weekly = self.weekData.get(date_str)

            vec = [
                (x["Close"]-x["Open"])/x["Open"]
                for x in hourly + daily + weekly
            ]

            rows.append([date] + vec)

        columns = ['Date'] + [f'f{i}' for i in range(68)]
        return pd.DataFrame(rows, columns=columns)

    def add_labels_to_context_df(self, context_df, threshold=0.0001):
        context_df['Date'] = pd.to_datetime(context_df['Date'])
        self.price_df['Date'] = pd.to_datetime(self.price_df['Date'])

        # Merge
        merged = context_df.merge(self.price_df[['Date', 'Open', 'Close']], on='Date', how='left')
        changes = (merged['Close'] - merged['Open']) / merged['Open']

        labels = np.zeros(len(changes), dtype=int) # HOLD
        labels[changes >= threshold] = Action.BUY.value
        labels[changes <= -threshold] = Action.SELL.value
        merged['Label'] = labels

        # Giữ nguyên thứ tự các feature
        feature_cols = [col for col in context_df.columns if col != 'Date']
        output_cols = ['Date'] + feature_cols + ['Label']
        return merged[output_cols]

    def add_reward_features(self, df):
        result = pd.DataFrame()
        result['Date'] = df['Date']
        base_reward = (df['Close'] - df['Open']) / df['Open']

        for i in range(self.num_experts):
            col = f'iteration{i}'
            action = df[col]

            accuracy = pd.Series(0, index=df.index)
            
            # Win
            accuracy[(action == Action.BUY.value) & (df['Close'] > df['Open'])] = 1
            accuracy[(action == Action.SELL.value) & (df['Close'] < df['Open'])] = 1

            # Loss
            accuracy[(action == Action.BUY.value) & (df['Close'] < df['Open'])] = -1
            accuracy[(action == Action.SELL.value) & (df['Close'] > df['Open'])] = -1

            # Tính reward theo accuracy
            reward = pd.Series(0.0, index=df.index)
            reward[accuracy == 1] = base_reward
            reward[accuracy == -1] = -base_reward

            result[f'e{i}_reward'] = reward
            result[f'e{i}_acc'] = accuracy
            result[f'e{i}_equity'] = (1.0 + reward).cumprod()
        return result
    
    def add_risk_features(self, reward_df, windows):
        reward_df = reward_df.sort_values('Date').reset_index(drop=True)
        result = pd.DataFrame()
        result['Date'] = reward_df['Date']

        for i in range(self.num_experts):
            reward_col = f'e{i}_reward'
            equity_col = f'e{i}_equity'

            # Rolling Volatility
            vol_col = f'e{i}_vol'
            result[vol_col] = reward_df[reward_col].rolling(window=windows, min_periods=1).std()
            result[vol_col].fillna(0.0, inplace=True)
            
            # rolling MDD
            mdd_col = f'e{i}_mdd'

            def window_mdd(x):
                peak = np.maximum.accumulate(x)
                drawdown = np.zeros_like(x)
                mask = peak > 0
                drawdown[mask] = (peak[mask] - x[mask]) / peak[mask]
                return drawdown.max()
            
            result[mdd_col] = reward_df[equity_col].rolling(window=windows, min_periods=1).apply(window_mdd, raw=True)
            result[mdd_col].fillna(0.0, inplace=True)

        return result
    
    def add_qvalues_features(self, q_values_df):
        rename_dict = {}
        for col in q_values_df.columns:
            if col.startswith("iteration"):
                new_col = col.replace("iteration", "e")
                rename_dict[col] = new_col

        q_values_df = q_values_df.rename(columns=rename_dict)
        q_values_df['Date'] = pd.to_datetime(q_values_df['Date'])
        return q_values_df

    def process_all_walks(self, phase="valid"):
        output_dir = f"{self.output_dir}/{phase}"
        os.makedirs(output_dir, exist_ok=True)

        for i in range(self.num_walks):
            action_path = os.path.join(self.ensemble_dir, f"walk{i}ensemble_{phase}.csv")
           
            if not os.path.exists(action_path):
                raise FileNotFoundError(f"Action file not found: {action_path}")
            
            action_df = pd.read_csv(action_path).iloc[2:]
            action_df['Date'] = pd.to_datetime(action_df['Date'])

            merged_df = pd.merge(action_df, self.price_df, on='Date', how='left')
            reward = self.add_reward_features(merged_df)
            reward.to_csv(os.path.join(output_dir, f"walk{i}_reward.csv"), index=False)
            
            risk = self.add_risk_features(reward, self.rolling_windows)
            risk.to_csv(os.path.join(output_dir, f"walk{i}_risk.csv"), index=False)

            qvalues_path = os.path.join(self.q_values_dir, f"{phase}/q_values_walk{i}.csv")

            if not os.path.exists(qvalues_path):
                raise FileNotFoundError(f"Q-values file not found: {qvalues_path}")
            
            q_values = pd.read_csv(qvalues_path).iloc[2:]
            q_values = self.add_qvalues_features(q_values)
            q_values.to_csv(os.path.join(output_dir, f"walk{i}_qvalues.csv"), index=False)

            # Save context features
            context = self.get_context_features(merged_df['Date'])
            labeled_context = self.add_labels_to_context_df(context)
            labeled_context.to_csv(os.path.join(output_dir, f"walk{i}_context.csv"), index=False)
