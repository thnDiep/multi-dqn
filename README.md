# Multi-DQN: an Ensemble of Deep Q-Learning Agents for Stock Market Forecasting

## Abstract

The stock market forecasting is one of the most challenging application of machine learning, as its historical data are naturally noisy and unstable. Most of the successful approaches act in a supervised manner, labeling training data as being of positive or negative moments of the market. However, training machine learning classifiers in such a way may suffer from over-fitting, since the market behavior depends on several external factors like other markets trends, political events, etc. In this paper, we aim at minimizing such problems by proposing an ensemble of reinforcement learning approaches which do not use annotations (i.e., market goes up or down) to learn, but rather learn how to maximize a return function over the training stage. In order to achieve this goal, we exploit a Q-learning agent trained several times with the same training data and investigate its ensemble behavior in important real-world stock markets. Experimental results in intraday trading indicate better performance than the conventional Buy-and-Hold strategy, which still behaves well in our setups. We also discuss qualitative and quantitative analyses of these results.

## Authors

-   Salvatore Carta
-   Anselmo Ferreira
-   Alessandro Sebastian Podda
-   Diego Reforgiato Recupero
-   Antonio Sanna

# Info

## Description

#### Core files:

-   **main.py**: Entry point of the application
-   **deepQTrading.py**: Organizes data and sets up the agents
-   **spEnv.py**: Environment for the agents
-   **mergedDataStructure.py**: Multi-timeframe data structure
-   **callback.py**: Module for logging and tracking results
-   **attention_network.py**: Module containing attention models:
    + GlobalFeatureAttention: Global attention on all features
    + LocalFeatureAttention: Local attention on each feature within timeframes
    + TimeFrameAttention: Attention on each timeframe

## Requirements

-   Python 3.6.8
-   Tensorflow 1.14.0: `pip install tensorflow==1.14.0`
-   Keras 2.3.1: `pip install keras==2.3.1`
-   Keras-RL 0.4.2: `pip install keras-rl==0.4.2`
-   OpenAI Gym 0.26.2: `pip install gym==0.26.2`
-   Pandas 0.25.3: `pip install pandas==0.25.3`
-   NumPy 1.19.5: `pip install numpy==1.19.5`

## Usage

The code supports two ensemble approaches:

### 1. Threshold-based Ensemble
This approach combines predictions from multiple models using a consensus threshold. If the number of models predicting a particular action exceeds the threshold, that action is selected.

Run with different models:
```
python -W ignore main.py dax original 0
python -W ignore main.py dax global_feature_atn 0
python -W ignore main.py dax local_feature_atn 0
python -W ignore main.py dax time_frame_atn 0
```

### 2. Mixture of Experts (MoE) Ensemble
This approach uses a MoE to dynamically weight the predictions from different expert models based on the current market conditions.

Run with different models:
```
python -W ignore main.py dax original 1
python -W ignore main.py dax global_feature_atn 1
python -W ignore main.py dax local_feature_atn 1
python -W ignore main.py dax time_frame_atn 1
```