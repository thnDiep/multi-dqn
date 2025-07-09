#Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import datetime

#This is the class call for the Agent which will perform the experiment
from expert.attention_network import ModelType
from utils.market_config import MARKET_CONFIG
from expert.deepQTrading import DeepQTrading
from router.moeTrading import MoeTrading

# sys.argv = ['main.py', 'dax', 'original', '1']

# Check input parameters
if len(sys.argv) < 4:
    print("Usage: python main.py <market> <modelName> <isMoe>")
    sys.exit(1)


#There are three actions possible in the stock market
#Hold(id 0): do nothing.
#Long(id 1): It predicts that the stock market value will raise at the end of the day. 
#So, the action performed in this case is buying at the beginning of the day and sell it at the end of the day (aka long).
#Short(id 2): It predicts that the stock market value will decrease at the end of the day.
#So, the action that must be done is selling at the beginning of the day and buy it at the end of the day (aka short). 
nb_actions = 3
isOnlyShort = 0

num_epochs = 100
market = sys.argv[1]
model_name = sys.argv[2]
isMoe = sys.argv[3]

# Check if market and model are valid
if market not in MARKET_CONFIG:
    print(f"Error: Market '{market}' is not supported")
    print(f"Supported markets: {', '.join(MARKET_CONFIG.keys())}")
    sys.exit(1)

if model_name not in ModelType.get_values():
    print(f"Error: Model '{model_name}' is not supported")
    print(f"Supported models: {', '.join(ModelType.get_values())}")
    sys.exit(1)

start_time = time.time()
if isMoe != '0':
    if isMoe == '1':
        moe_model = "flat"
    elif isMoe == '2':
        moe_model = "2d"
    else:
        print("Invalid MoE model")
        sys.exit(1)

    config = {
        'lr': 5e-4,
        # 'weight_decay': 1e-3,
        'weight_decay': 0.0,
        'label_smoothing': 0.0,
        # 'lambda_entropy': 0.1
        'lambda_entropy': 0.0
    }

    model = MoeTrading(
        market=market,
        model_name=model_name,
        moe_model_type=moe_model,  # hoặc "flat"
        num_epochs=num_epochs,
        **config  # unpack grid search hyperparams
    )

    model.run()
    # model.end()
else:
    model = DeepQTrading(
        market=market,
        model_name=model_name,
        explorations=[(0.2,num_epochs)],
        trainSize=datetime.timedelta(days=360*5),
        validationSize=datetime.timedelta(days=30*6),
        testSize=datetime.timedelta(days=30*6),
        nbActions=nb_actions,
        isOnlyShort=isOnlyShort,
        )
   
    # model.run()
    model.end()

end_time = time.time()
training_time = end_time - start_time
print(f"\nThời gian training: {training_time:.2f} giây ({training_time/60:.2f} phút)")

