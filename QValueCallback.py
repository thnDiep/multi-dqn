from keras.callbacks import Callback
import os
import pandas as pd
import numpy as np
from datetime import datetime

class QValueCallback(Callback):
    def __init__(self, output_dir='./Output/q_values', phase='train', walk=0):
        super(QValueCallback, self).__init__()
        self.output_dir = os.path.join(output_dir, phase)
        self.phase = phase  # 'train', 'valid', or 'test'
        self.walk = walk
        self.current_epoch = 0
        self.q_values_df = None 
        self.env = None
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def reset(self, walk, dates):
        """Reset DataFrame for new walk"""
        self.walk = walk
        self.current_epoch = 0
        
        # Initialize DataFrame with dates as index
        self.q_values_df = pd.DataFrame(index=dates)
        self.q_values_df.index.name = 'Date'
                
        
    def on_step_end(self, step, logs):
        # Get observation from logs
        observation = logs.get('observation')
        
        # Get date from SpEnv through callback
        if hasattr(self, 'env') and hasattr(self.env, 'history'):
            current_date = self.env.history[self.env.currentObservation]['Date']
            
            if observation is not None:
                state = self.model.memory.get_recent_state(observation)
                q_values = self.model.compute_q_values(state)
                if q_values is not None:
                    # Save values to DataFrame
                    for i, value in enumerate(q_values):
                        col_name = f'iteration{self.current_epoch}_q{i}'
                        self.q_values_df.at[current_date, col_name] = value
                
    def set_epoch(self, epoch):
        """Set current epoch number"""
        self.current_epoch = epoch
        
    def set_env(self, env):
        """Set environment reference"""
        self.env = env
        
    def save_file(self):
        """Save DataFrame to file"""
        if self.q_values_df is not None:
            # Fill zeros for first two rows
            if len(self.q_values_df) >= 2:
                self.q_values_df.iloc[0:2] = self.q_values_df.iloc[0:2].fillna(0)
            
            # Save to file with walk name
            filepath = os.path.join(self.output_dir, f'q_values_walk{self.walk}.csv')
            self.q_values_df.to_csv(filepath)
        