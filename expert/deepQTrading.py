# Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

import os
import time
import datetime
import pandas as pd
from math import floor
import keras.backend as K
from keras.optimizers import Adam

# RL Agent
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

from environments.spEnv import SpEnv
from expert.attention_network import build_model
from evaluation.evaluation import Evaluation
from utils.market_config import get_market_config
from utils.callback import ValidationCallback, QValueCallback, AttentionCallback


class DeepQTrading:

    # Class constructor
    # model: Keras model considered
    # Explorations is a vector containing (i) probability of random predictions; (ii) how many epochs will be
    # runned by the algorithm (we run the algorithm several times-several iterations)  
    # trainSize: size of the training set
    # validationSize: size of the validation set
    # testSize: size of the testing set
    # outputFile: name of the file to print results
    # begin: Initial date
    # end: final date
    # nbActions: number of decisions (0-Hold 1-Long 2-Short)
    # nOutput is the number of walks. We are doing 5 walks.
    # operationCost: Price for the transaction (we set they are free)
    def __init__(self, market, model_name, explorations, trainSize, validationSize, testSize, nbActions, isOnlyShort):
        # Prepare model and market config
        self.model_name = model_name
        self.model, self.custom_objects = build_model(model_name)

        self.market = market
        self.market_config = get_market_config(market)

        # Prepare folder paths
        training_dir = f"./Output/training/{market}/{model_name}"
        self.training_file_path = f"{training_dir}/walks"

        self.ensemble_dir = f"./Output/ensemble/{market}/{model_name}"

        self.result_dir = f"./Output/results/{market}/"

        model_dir = f"./Output/models/{market}"
        self.weights_file = f"{model_dir}/q-{model_name}.weights"

        self.q_values_dir = f"./Output/q_values/{market}/{model_name}"
        self.atn_dir = f"./Output/attentions/{market}/{model_name}"

        self.final_decision_dir = f"./Output/final_decision/{market}/{model_name}"

        os.makedirs(training_dir, exist_ok=True)
        os.makedirs(self.ensemble_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(self.q_values_dir, exist_ok=True)
        os.makedirs(self.final_decision_dir, exist_ok=True)

        self.isOnlyShort = isOnlyShort

        # Define the policy, explorations, actions and model as received by parameters
        self.policy = EpsGreedyQPolicy()
        self.explorations = explorations
        self.nbActions = nbActions

        # Define the memory
        self.memory = SequentialMemory(limit=10000, window_length=1)

        # Instantiate the agent with parameters received
        self.agent = DQNAgent(model=self.model,
                              policy=self.policy,
                              nb_actions=self.nbActions,
                              memory=self.memory,
                              nb_steps_warmup=200,
                              target_model_update=1e-1,
                              enable_double_dqn=True,
                              enable_dueling_network=True,
                              custom_model_objects=self.custom_objects)

        # Compile the agent with the adam optimizer and with the mean absolute error metric
        self.agent.compile(Adam(lr=1e-3), metrics=['mae'])

        # If test only, load trained weights
        self.agent.save_weights(self.weights_file, overwrite=True)

        # Define the current starting point as the initial date
        self.currentStartingPoint = self.market_config["start_date"]

        # Define the training, validation and testing size as informed by the call
        # Train: 5 years
        # Validation: 6 months
        # Test: 6 months
        self.trainSize = trainSize
        self.validationSize = validationSize
        self.testSize = testSize

        # The walk size is simply summing up the train, validation and test sizes
        self.walkSize = trainSize + validationSize + testSize

        # Define the ending point as the final date (January 1st of 2010)
        self.endingPoint = self.market_config["end_date"]

        # Read the hourly dataset
        # We join data from different files
        # Here hour data is read
        self.dates = pd.read_csv(f'./datasets/{self.market}Hour.csv')
        self.sp = pd.read_csv(f'./datasets/{self.market}Hour.csv')
        # Convert the pandas format to date and time format
        self.sp['Datetime'] = pd.to_datetime(self.sp['Date'] + ' ' + self.sp['Time'])
        # Set an index to Datetime on the pandas loaded dataset. Registers will be indexes through these values
        self.sp = self.sp.set_index('Datetime')
        # Drop Time and Date from the Dataset
        self.sp = self.sp.drop(['Time', 'Date'], axis=1)
        # Just the index considering date and time will be important, because date and time will be used to define the train,
        # validation and test for each walk
        self.sp = self.sp.index

        # Receives the operation cost, which is 0
        # Operation cost is the cost for long and short. It is defined as zero
        self.operationCost = 0

        # Call the callback for training, validation and test in order to show results for each episode
        self.trainer = ValidationCallback()
        self.validator = ValidationCallback()
        self.tester = ValidationCallback()

        # Add QValue callbacks with separate output_dir for each phase and walk
        self.train_q_callback = QValueCallback(output_dir=self.q_values_dir, phase='train')
        self.valid_q_callback = QValueCallback(output_dir=self.q_values_dir, phase='valid')
        self.test_q_callback = QValueCallback(output_dir=self.q_values_dir, phase='test')

        if self.model_name == "time_frame_atn":
            os.makedirs(self.atn_dir, exist_ok=True)
            self.test_atn_callback = AttentionCallback(output_dir=self.atn_dir, phase='test')
        else:
            self.test_atn_callback = None
        # self.test_atn_callback = None

    def run(self):
        # Initiates the environments,
        trainEnv = validEnv = testEnv = " "

        iteration = -1

        # While we did not pass through all the dates (i.e., while all the walks were not finished)
        # walk size is train+validation+test size
        # currentStarting point begins with begin date
        while (self.currentStartingPoint + self.walkSize <= self.endingPoint):

            # Iteration is the current walk
            iteration += 1

            # Initiate the output file với tên file khác nhau cho test only
            self.training_file = open(f"{self.training_file_path}{str(iteration + 1)}.csv", "w+")
            # write the first row of the csv
            self.training_file.write(
                "Iteration," +
                "trainAccuracy," +
                "trainCoverage," +
                "trainReward," +
                "trainLong%," +
                "trainShort%," +
                "trainLongAcc," +
                "trainShortAcc," +
                "trainLongPrec," +
                "trainShortPrec," +

                "validationAccuracy," +
                "validationCoverage," +
                "validationReward," +
                "validationLong%," +
                "validationShort%," +
                "validationLongAcc," +
                "validationShortAcc," +
                "validLongPrec," +
                "validShortPrec," +

                "testAccuracy," +
                "testCoverage," +
                "testReward," +
                "testLong%," +
                "testShort%," +
                "testLongAcc," +
                "testShortAcc," +
                "testLongPrec," +
                "testShortPrec\n")

            # Empty the memory and agent
            del (self.memory)
            del (self.agent)

            # Define the memory and agent
            # Memory is Sequential
            self.memory = SequentialMemory(limit=10000, window_length=1)
            # Agent is initiated as passed through parameters
            self.agent = DQNAgent(model=self.model,
                                  policy=self.policy,
                                  nb_actions=self.nbActions,
                                  memory=self.memory,
                                  nb_steps_warmup=200,
                                  target_model_update=1e-1,
                                  enable_double_dqn=True,
                                  enable_dueling_network=True,
                                  custom_model_objects=self.custom_objects)
            # Compile the agent with Adam initialization
            self.agent.compile(Adam(lr=1e-3), metrics=['mae'])

            # Load the weights saved before in a random way if it is the first time
            self.agent.load_weights(self.weights_file)

            ########################################TRAINING STAGE########################################################

            # The TrainMinLimit will be loaded as the initial date at the beginning, and will be updated later.
            # If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date
            trainMinLimit = None
            while (trainMinLimit is None):
                try:
                    trainMinLimit = self.sp.get_loc(self.currentStartingPoint)
                except:
                    self.currentStartingPoint += datetime.timedelta(0, 0, 0, 0, 0, 1, 0)

            # The TrainMaxLimit will be loaded as the interval between the initial date plus the training size.
            # If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date
            trainMaxLimit = None
            while (trainMaxLimit is None):
                try:
                    trainMaxLimit = self.sp.get_loc(self.currentStartingPoint + self.trainSize)
                except:
                    self.currentStartingPoint += datetime.timedelta(0, 0, 0, 0, 0, 1, 0)

            ########################################VALIDATION STAGE#######################################################
            # The ValidMinLimit will be loaded as the next element of the TrainMax limit
            validMinLimit = trainMaxLimit + 1

            # The ValidMaxLimit will be loaded as the interval after the begin + train size +validation size
            # If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date
            validMaxLimit = None
            while (validMaxLimit is None):
                try:
                    validMaxLimit = self.sp.get_loc(self.currentStartingPoint + self.trainSize + self.validationSize)
                except:
                    self.currentStartingPoint += datetime.timedelta(0, 0, 0, 0, 0, 1, 0)

            ########################################TESTING STAGE######################################################## 
            # The TestMinLimit will be loaded as the next element of ValidMaxlimit
            testMinLimit = validMaxLimit + 1

            # The testMaxLimit will be loaded as the interval after the begin + train size +validation size + Testsize
            # If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date
            testMaxLimit = None
            while (testMaxLimit is None):
                try:
                    testMaxLimit = self.sp.get_loc(
                        self.currentStartingPoint + self.trainSize + self.validationSize + self.testSize)
                except:
                    self.currentStartingPoint += datetime.timedelta(0, 0, 0, 0, 0, 1, 0)

            # Separate the Validation and testing data according to the limits found before
            # Prepare the training and validation files for saving them later
            ensambleTrain = pd.DataFrame(
                index=self.dates[trainMinLimit:trainMaxLimit].ix[:, 'Date'].drop_duplicates().tolist())
            ensambleValid = pd.DataFrame(
                index=self.dates[validMinLimit:validMaxLimit].ix[:, 'Date'].drop_duplicates().tolist())
            ensambleTest = pd.DataFrame(
                index=self.dates[testMinLimit:testMaxLimit].ix[:, 'Date'].drop_duplicates().tolist())

            # Put the name of the index for validation and testing
            ensambleTrain.index.name = 'Date'
            ensambleValid.index.name = 'Date'
            ensambleTest.index.name = 'Date'

            # Reset QValue callbacks khi bắt đầu walk mới
            self.train_q_callback.reset(iteration, self.dates[trainMinLimit:trainMaxLimit].ix[:,
                                                   'Date'].drop_duplicates().tolist())
            self.valid_q_callback.reset(iteration, self.dates[validMinLimit:validMaxLimit].ix[:,
                                                   'Date'].drop_duplicates().tolist())
            self.test_q_callback.reset(iteration,
                                       self.dates[testMinLimit:testMaxLimit].ix[:, 'Date'].drop_duplicates().tolist())

            if self.test_atn_callback is not None:
                self.test_atn_callback.reset(iteration, self.dates[testMinLimit:testMaxLimit].ix[:,
                                                        'Date'].drop_duplicates().tolist())

            # Explorations are epochs considered, or how many times the agent will play the game.
            for eps in self.explorations:
                # policy will be 0.2, so the randomness of predictions (actions) will happen with 20% of probability
                self.policy.eps = eps[0]

                # there will be 100 iterations (epochs), or eps[1])
                for i in range(0, eps[1]):
                    start_time = time.time()

                    # Set epoch number cho callbacks
                    self.train_q_callback.set_epoch(i)
                    self.valid_q_callback.set_epoch(i)
                    self.test_q_callback.set_epoch(i)

                    if self.test_atn_callback is not None:
                        self.test_atn_callback.set_epoch(i)

                    del (trainEnv)
                    # Define the training environment with its callback
                    trainEnv = SpEnv(self.market, operationCost=self.operationCost, minLimit=trainMinLimit,
                                     maxLimit=trainMaxLimit, callback=self.trainer, isOnlyShort=self.isOnlyShort,
                                     ensamble=ensambleTrain, columnName="iteration" + str(i))
                    # Reset the callback
                    self.trainer.reset()
                    # Set environment cho QValueCallback
                    self.train_q_callback.set_env(trainEnv)
                    # Reset the training environment
                    trainEnv.resetEnv()

                    # Train the agent
                    self.agent.fit(trainEnv, nb_steps=floor(self.trainSize.days - self.trainSize.days * 0.2),
                                   visualize=False, verbose=0, callbacks=[self.train_q_callback])
                    # Get the info from the train callback
                    (_, trainCoverage, trainAccuracy, trainReward, trainLongPerc, trainShortPerc, trainLongAcc,
                     trainShortAcc, trainLongPrec, trainShortPrec) = self.trainer.getInfo()
                    # Print Callback values on the screen
                    print(
                        str(i) + " TRAIN:  acc: " + str(trainAccuracy) + " cov: " + str(trainCoverage) + " rew: " + str(
                            trainReward))

                    del (validEnv)
                    validEnv = SpEnv(self.market, operationCost=self.operationCost, minLimit=validMinLimit,
                                     maxLimit=validMaxLimit, callback=self.validator, isOnlyShort=self.isOnlyShort,
                                     ensamble=ensambleValid, columnName="iteration" + str(i))
                    # Reset the callback
                    self.validator.reset()
                    # Set environment cho QValueCallback
                    self.valid_q_callback.set_env(validEnv)
                    # Reset the validation environment
                    validEnv.resetEnv()
                    # Test the agent on validation data
                    self.agent.test(validEnv,
                                    nb_episodes=floor(self.validationSize.days - self.validationSize.days * 0.2),
                                    visualize=False, verbose=0, callbacks=[self.valid_q_callback])
                    # Get the info from the validation callback
                    (_, validCoverage, validAccuracy, validReward, validLongPerc, validShortPerc, validLongAcc,
                     validShortAcc, validLongPrec, validShortPrec) = self.validator.getInfo()
                    # Print callback values on the screen
                    print(
                        str(i) + " VALID:  acc: " + str(validAccuracy) + " cov: " + str(validCoverage) + " rew: " + str(
                            validReward))

                    del (testEnv)
                    testEnv = SpEnv(self.market, operationCost=self.operationCost, minLimit=testMinLimit,
                                    maxLimit=testMaxLimit, callback=self.tester, isOnlyShort=self.isOnlyShort,
                                    ensamble=ensambleTest, columnName="iteration" + str(i))
                    # Reset the callback
                    self.tester.reset()
                    # Set environment cho QValueCallback
                    self.test_q_callback.set_env(testEnv)
                    test_callbacks = [self.test_q_callback]

                    if self.test_atn_callback is not None:
                        self.test_atn_callback.set_env(testEnv)
                        test_callbacks.append(self.test_atn_callback)

                    # Reset the testing environment
                    testEnv.resetEnv()
                    # Test the agent on testing data
                    self.agent.test(testEnv, nb_episodes=floor(self.testSize.days - self.testSize.days * 0.2),
                                    visualize=False, verbose=0, callbacks=test_callbacks)
                    # Get the info from the testing callback
                    (_, testCoverage, testAccuracy, testReward, testLongPerc, testShortPerc, testLongAcc, testShortAcc,
                     testLongPrec, testShortPrec) = self.tester.getInfo()
                    # Print callback values on the screen
                    print(str(i) + " TEST:  acc: " + str(testAccuracy) + " cov: " + str(testCoverage) + " rew: " + str(
                        testReward))

                    # write the walk data on the text file
                    self.training_file.write(
                        str(i) + "," +
                        str(trainAccuracy) + "," +
                        str(trainCoverage) + "," +
                        str(trainReward) + "," +
                        str(trainLongPerc) + "," +
                        str(trainShortPerc) + "," +
                        str(trainLongAcc) + "," +
                        str(trainShortAcc) + "," +
                        str(trainLongPrec) + "," +
                        str(trainShortPrec) + "," +

                        str(validAccuracy) + "," +
                        str(validCoverage) + "," +
                        str(validReward) + "," +
                        str(validLongPerc) + "," +
                        str(validShortPerc) + "," +
                        str(validLongAcc) + "," +
                        str(validShortAcc) + "," +
                        str(validLongPrec) + "," +
                        str(validShortPrec) + "," +

                        str(testAccuracy) + "," +
                        str(testCoverage) + "," +
                        str(testReward) + "," +
                        str(testLongPerc) + "," +
                        str(testShortPerc) + "," +
                        str(testLongAcc) + "," +
                        str(testShortAcc) + "," +
                        str(testLongPrec) + "," +
                        str(testShortPrec) + "\n")

                    end_time = time.time()
                    epoch_time = end_time - start_time
                    print(f"\nThời gian 1 epoch: {epoch_time:.2f} giây ({epoch_time / 60:.2f} phút)")
                    print(" ")

            # Close the file
            self.training_file.close()

            # For the next walk, the current starting point will be the current starting point + the test size
            # It means that, for the next walk, the training data will start 6 months after the training data of
            # the previous walk
            self.currentStartingPoint += self.testSize

            # Save Q-values before moving to next walk
            self.train_q_callback.save_file()
            self.valid_q_callback.save_file()
            self.test_q_callback.save_file()

            if self.test_atn_callback is not None:
                self.test_atn_callback.save_file()

            # Write validation and Testing data into files với tên file khác nhau cho test only

            ensambleTrain.to_csv(f"{self.ensemble_dir}/walk{iteration}ensemble_train.csv")
            ensambleValid.to_csv(f"{self.ensemble_dir}/walk{iteration}ensemble_valid.csv")
            ensambleTest.to_csv(f"{self.ensemble_dir}/walk{iteration}ensemble_test.csv")

    # Function to end the Agent
    def end(self):
        """End the training process and create an evaluation report"""

        evaluation = Evaluation(
            model_name=self.model_name,
            market=self.market,
            input_dir=self.ensemble_dir,
            result_dir=self.result_dir,
            final_decision_dir=self.final_decision_dir,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
        )

        evaluation.plot_results(ensemble_type="ensemble")

        print(f"Evaluation completed for {self.market}. Check the Output directory for results.")
