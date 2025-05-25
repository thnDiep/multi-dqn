#Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

#os library is used to define the GPU to be used by the code, needed only in cerain situations (Better not to use it, use only if the main gpu is Busy)
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import tensorflow as tf
import time  # Thêm import time

#This is the class call for the Agent which will perform the experiment
from deepQTrading import DeepQTrading

#Date library to manipulate time in the source code
import datetime

#Keras library to define the NN to be used
from keras.models import Sequential

#Layers used in the NN considered
from keras.layers import Dense, Activation, Flatten

#Activation Layers used in the source code
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU

#Optimizer used in the NN
from keras.optimizers import Adam

#Libraries used for the Agent considered
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

from attention_network import ModelType
from market_config import get_market_config, MARKET_CONFIG

#Library used for showing the exception in the case of error 
import sys

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))

start_time = time.time()  # Thêm đo thời gian bắt đầu

# Check input parameters
if len(sys.argv) < 5:
    print("Usage: python main.py <numberOfActions> <isOnlyShort> <market> <modelName>")
    print(f"Supported markets: {', '.join(MARKET_CONFIG.keys())}")
    print(f"Supported models: {', '.join(ModelType.get_values())}")
    sys.exit(1)

#There are three actions possible in the stock market
#Hold(id 0): do nothing.
#Long(id 1): It predicts that the stock market value will raise at the end of the day. 
#So, the action performed in this case is buying at the beginning of the day and sell it at the end of the day (aka long).
#Short(id 2): It predicts that the stock market value will decrease at the end of the day.
#So, the action that must be done is selling at the beginning of the day and buy it at the end of the day (aka short). 
nb_actions = int(sys.argv[1])
isOnlyShort = sys.argv[2]==1
market = sys.argv[3]
model_name = sys.argv[4]

# Check if market and model are valid
if market not in MARKET_CONFIG:
    print(f"Error: Market '{market}' is not supported")
    print(f"Supported markets: {', '.join(MARKET_CONFIG.keys())}")
    sys.exit(1)

if model_name not in ModelType.get_values():
    print(f"Error: Model '{model_name}' is not supported")
    print(f"Supported models: {', '.join(ModelType.get_values())}")
    sys.exit(1)

#Define the DeepQTrading class with the following parameters:
#explorations: 0.2 operations are random, and 100 epochs.
#in this case, epochs parameter is used because the Agent acts on daily basis, so its better to repeat the experiments several
#times so, its defined that each epoch will work on the data from training, validation and testing.
#trainSize: the size of the train data gotten from the dataset, we are setting 5 stock market years, or 1800 days
#validationSize: the size of the validation data gotten from dataset, we are setting 6 stock market months, or 180 days
#testSize: the size of the testing data gotten from dataset, we are setting 6 stock market months, or 180 days
#outputFile: where the results will be written
#begin: where the walks will start from. We are defining January 1st of 2010
#end: where the walks will finish. We are defining February 22nd of 2019
#nOutput:number of walks
dqt = DeepQTrading(
    model_name=model_name,
    market=market,
    explorations=[(0.2,1)],
    trainSize=datetime.timedelta(days=360*5),
    validationSize=datetime.timedelta(days=30*6),
    testSize=datetime.timedelta(days=30*6),
    nbActions=nb_actions,
    isOnlyShort=isOnlyShort,
    )

dqt.run()

end_time = time.time()
training_time = end_time - start_time
print(f"\nThời gian training: {training_time:.2f} giây ({training_time/60:.2f} phút)")

dqt.end()

