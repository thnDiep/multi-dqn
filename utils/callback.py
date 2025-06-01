#Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.
#Callbacks are functions used to give a feedback about each epoch calculated metrics
import os
import numpy as np
import pandas as pd
from keras import backend as K
from rl.callbacks import Callback


class ValidationCallback(Callback):

    def __init__(self):
        #Initially, the metrics are zero
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        self.shortAcc =0
        self.longAcc =0
        self.longPrec =0
        self.shortPrec =0
        self.marketRise =0
        self.marketFall =0

    def reset(self):
        #The metrics are also zero when the epoch ends
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        self.shortAcc =0
        self.longAcc =0
        self.longPrec =0
        self.shortPrec =0
        self.marketRise =0
        self.marketFall =0
        
    #all information is given by the environment: action, reward and market
    #Then, when the episode ends, metrics are calculated
    def on_episode_end(self, action, reward, market):
        
        #After the episode ends, increments the episodes 
        self.episodes+=1

        #Increments the reward
        self.rewardSum+=reward

        #If the action is not a hold, there is coverage because the agent decided 
        self.coverage+=1 if (action != 0) else 0

        #increments the accuracy if the reward is positive (we have a hit)
        self.accuracy+=1 if (reward >= 0 and action != 0) else 0
        
        #Increments the counter for short if the action is a short (id 2)
        self.short +=1 if(action == 2) else 0
        
        #Increments the counter for long if the action is a long (id 1)
        self.long +=1 if(action == 1) else 0
        
        #We will also calculate the accuracy for a given action. Here, it increments
        #the accuracy for short if the action is short and the reward is positive
        self.shortAcc +=1 if(action == 2 and reward >=0) else 0
        
        #Increments the accuracy for long if the action is long and the reward is positive
        self.longAcc +=1 if(action == 1 and reward >=0) else 0
        
        #If the market increases, increments the marketRise variable. If the prediction is 1 (long), increments the precision for long
        if(market>0):
            self.marketRise+=1
            self.longPrec+=1 if(action == 1) else 0

        #If market decreases, increments the marketFall. If the prediction is 2 (short), increments the precision for short   
        elif(market<0):
            self.marketFall+=1
            self.shortPrec+=1 if(action == 2) else 0

    #Function to show the metrics of the episode  
    def getInfo(self):
        #Start setting them to zero
        acc = 0
        cov = 0
        short = 0
        long = 0
        longAcc = 0
        shortAcc = 0
        longPrec = 0
        shortPrec = 0
        
        #If there is coverage, we will calculate the accuracy only related to when decisions were made. 
        #In other words, we dont calculate accuracy for hold operations
        if self.coverage > 0:
            acc = self.accuracy/self.coverage
        
        #Now, we calculate the mean coverage, short and long operations from the episodes
        if self.episodes > 0:
            cov = self.coverage/self.episodes
            short = self.short/self.episodes
            long = self.long/self.episodes

        #Calculate the mean accuracy for short operations. 
        #That is, the number of total short correctly predicted (self.shortAcc) 
        #divided by the total of shorts predicted (self.short)
        if self.short > 0:
            shortAcc = self.shortAcc/self.short
        
        #Calculate the mean accuracy for long operations. 
        #That is, the number of total short correctly predicted (long.shortAcc) 
        #divided by the total of longs predicted (long.short) 
        if self.long > 0:
            longAcc = self.longAcc/self.long

        if self.marketRise > 0:
            longPrec = self.longPrec/self.marketRise

        if self.marketFall > 0:
            shortPrec = self.shortPrec/self.marketFall

        #Returns the metrics to the user    
        return self.episodes,cov,acc,self.rewardSum,long,short,longAcc,shortAcc,longPrec,shortPrec


class QValueCallback(Callback):
    def __init__(self, output_dir='./Output/q_values', phase='train', walk=0):
        super(QValueCallback, self).__init__()
        self.output_dir = os.path.join(output_dir, phase)
        self.phase = phase  # 'train', 'valid', or 'test'
        self.walk = walk
        self.current_epoch = 0
        self.q_values_df = None 
        self.attention_df = None 
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
                    # Save Q-values to DataFrame
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
        """Save DataFrames to files"""
        if self.q_values_df is not None:
            # Fill zeros for first two rows
            if len(self.q_values_df) >= 2:
                self.q_values_df.iloc[0:2] = self.q_values_df.iloc[0:2].fillna(0)
            
        # Save Q-values
        q_values_path = os.path.join(self.output_dir, f'q_values_walk{self.walk}.csv')
        self.q_values_df.to_csv(q_values_path)
        

class AttentionCallback(Callback):
    def __init__(self, output_dir='./Output/attentions', phase='test', walk=0):
        super(AttentionCallback, self).__init__()
        self.output_dir = os.path.join(output_dir, phase)
        self.phase = phase  # 'train', 'valid', or 'test'
        self.walk = walk
        self.current_epoch = 0
        self.atn_df = None
        self.env = None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def reset(self, walk, dates):
        """Reset DataFrame for new walk"""
        self.walk = walk
        self.current_epoch = 0

        # Initialize DataFrame with dates as index
        self.atn_df = pd.DataFrame(index=dates)
        self.atn_df.index.name = 'Date'

    def on_step_end(self, step, logs):
        # Get observation from logs
        observation = logs.get('observation')

        # Get date from SpEnv through callback
        if hasattr(self, 'env') and hasattr(self.env, 'history'):
            current_date = self.env.history[self.env.currentObservation]['Date']

            if observation is not None:
                state = self.model.memory.get_recent_state(observation)
                attention_layer = self.model.model.layers[2]
                state_reshaped = np.reshape(state, (1, 1, 1, 68))
                get_attention_weights = K.function([self.model.model.input],
                                            [attention_layer.attention_weights])
                attention_weights = get_attention_weights([state_reshaped])[0]

                # Lưu attention weights vào DataFrame
                self.atn_df.at[current_date, f'iteration{self.current_epoch}_hour'] = float(attention_weights[0, 0])
                self.atn_df.at[current_date, f'iteration{self.current_epoch}_day'] = float(attention_weights[0, 1])
                self.atn_df.at[current_date, f'iteration{self.current_epoch}_week'] = float(attention_weights[0, 2])

    def set_epoch(self, epoch):
        """Set current epoch number"""
        self.current_epoch = epoch

    def set_env(self, env):
        """Set environment reference"""
        self.env = env

    def save_file(self):
        """Save DataFrames to files"""
        if self.atn_df is not None:
            # Fill zeros for first two rows
            if len(self.atn_df) >= 2:
                self.atn_df.iloc[0:2] = self.atn_df.iloc[0:2].fillna(0)

        attention_path = os.path.join(self.output_dir, f'attention_weights_walk{self.walk}.csv')
        self.atn_df.to_csv(attention_path)