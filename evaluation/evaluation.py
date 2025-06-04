import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.market_config import get_market_config
from evaluation.trading import IndayTrading, RealisticTrading

class Evaluation:
    def __init__(self, model_name, market, input_dir, result_dir, final_decision_dir, stop_loss_pct=0.02, take_profit_pct=0.04):
        self.model_name = model_name
        self.market = market
        self.market_data = pd.read_csv(f"./datasets/{market}Day.csv", index_col='Date')
        self.market_data.index = pd.to_datetime(self.market_data.index)
        self.num_walks = get_market_config(market)["num_walks"]

        self.input_dir = input_dir
        self.result_dir = result_dir
        self.final_decision_dir = final_decision_dir
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.final_action = "ensemble"
        
    def full_ensemble(self, df):
        """Calculate ensemble with 100% consensus"""
        m1 = df.eq(1).all(axis=1)
        m2 = df.eq(2).all(axis=1)
        local_df = df.copy()
        local_df[self.final_action] = np.select([m1, m2], [1, 2], 0)
        local_df = local_df.drop(local_df.columns.difference([self.final_action]), axis=1)
        return local_df

    def perc_ensemble(self, df, thr=0.7):
        """Calculate ensemble with specific consensus threshold"""
        c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
        c2 = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
        return pd.DataFrame(np.select([c1, c2], [1, 2], 0), index=df.index, columns=[self.final_action])

    def evaluate(self, consensus_threshold=0, type="test"):
        """
        Evaluate model performance with both inday trading and realistic trading strategies
        """
        # Initialize results storage
        inday_trading = IndayTrading(self.market_data)
        realistic_trading = RealisticTrading(self.market_data, self.stop_loss_pct, self.take_profit_pct)

        # Process each walk file once
        for j in range(self.num_walks):
            df = pd.read_csv(f"{self.input_dir}/walk{j}ensemble_{type}.csv", index_col='Date')
            df.index = pd.to_datetime(df.index)
            
            # Perform ensemble
            if consensus_threshold == 0:
                df = self.full_ensemble(df)
            else:
                df = self.perc_ensemble(df, consensus_threshold)
            
            df.to_csv(f"{self.final_decision_dir}/walk{j}_{type}_{consensus_threshold}.csv")

            inday_trading.trading_for_each_walk(df, j) # Evaluate inday trading
            realistic_trading.trading_for_each_walk(df, j) # Evaluate realistic trading
        
        # Get results for both strategies
        inday_values, inday_columns = inday_trading.get_total_walk_result()
        realistic_values, realistic_columns = realistic_trading.get_total_walk_result()
        
        realistic_trading.plot_equity_curve(f"{self.result_dir}/equity_curve_{consensus_threshold}.pdf")

        return {
            'inday': {'values': inday_values, 'columns': inday_columns},
            'realistic': {'values': realistic_values, 'columns': realistic_columns}
        }
    
    def plot_ensemble_results(self):
        """Plot ensemble results tables with different thresholds"""
        pdf_inday = PdfPages(f"{self.result_dir}/inday_trading.pdf")
        pdf_realistic = PdfPages(f"{self.result_dir}/realistic_trading.pdf")

        
        thresholds = [0, 0.9, 0.8, 0.7, 0.6]
        titles = ["FULL ENSEMBLE", "90% ENSEMBLE", "80% ENSEMBLE", "70% ENSEMBLE", "60% ENSEMBLE"]
        
        for threshold, title in zip(thresholds, titles):
            results = self.evaluate(threshold, "valid")
            inday_valid = results['inday']

            results = self.evaluate(threshold, "test")
            inday_test = results['inday']
            realistic_test = results['realistic']

            plt.figure(figsize=(12, 5))
            
            # Validation results
            plt.subplot(1, 2, 1)
            plt.axis('off')
            t = plt.table(cellText=inday_valid['values'], colLabels=inday_valid['columns'], fontsize=30, loc='center')
            t.auto_set_font_size(False)
            t.set_fontsize(6)
            plt.title("Valid")
            
            # Test results
            plt.subplot(1, 2, 2)
            plt.axis('off')
            t = plt.table(cellText=inday_test['values'], colLabels=inday_test['columns'], fontsize=30, loc='center')
            t.auto_set_font_size(False)
            t.set_fontsize(6)
            plt.title("Test")
            
            plt.suptitle(title)
            pdf_inday.savefig()


            plt.figure(figsize=(12, 5))
            plt.axis('off')
            t = plt.table(cellText=realistic_test['values'], colLabels=realistic_test['columns'], fontsize=30, loc='center')
            t.auto_set_font_size(False)
            t.set_fontsize(6)
            plt.title("Test")
            
            plt.suptitle(title)
            pdf_realistic.savefig()

        pdf_inday.close()
        pdf_realistic.close()

    def evaluate_for_moe(self, type="test"):
        """
        Evaluate model performance with both inday trading and realistic trading strategies
        """
        # Initialize results storage
        inday_trading = IndayTrading(self.market_data)
        realistic_trading = RealisticTrading(self.market_data, self.stop_loss_pct, self.take_profit_pct)

        # Process each walk file once
        for j in range(self.num_walks):
            df = pd.read_csv(f"{self.input_dir}/walk{j}_{type}.csv", index_col='Date')
            df.index = pd.to_datetime(df.index)
          
            inday_trading.trading_for_each_walk(df, j) # Evaluate inday trading
            realistic_trading.trading_for_each_walk(df, j) # Evaluate realistic trading
        
        # Get results for both strategies
        inday_values, inday_columns = inday_trading.get_total_walk_result()
        realistic_values, realistic_columns = realistic_trading.get_total_walk_result()
        
        realistic_trading.plot_equity_curve(f"{self.result_dir}/equity_curve.pdf")

        return {
            'inday': {'values': inday_values, 'columns': inday_columns},
            'realistic': {'values': realistic_values, 'columns': realistic_columns}
        }

    def plot_moe_results(self):
        """Plot MOE results tables with different thresholds"""
        pdf_inday = PdfPages(f"{self.result_dir}/results_moe.pdf")

        results = self.evaluate_for_moe("valid")
        inday_valid = results['inday']

        results = self.evaluate_for_moe("test")
        inday_test = results['inday']
        realistic_test = results['realistic']

        plt.figure(figsize=(12, 5))
            
        # Validation results
        plt.subplot(1, 2, 1)
        plt.axis('off')
        t = plt.table(cellText=inday_valid['values'], colLabels=inday_valid['columns'], fontsize=30, loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(6)
        plt.title("Valid")
        
        # Test results
        plt.subplot(1, 2, 2)
        plt.axis('off')
        t = plt.table(cellText=inday_test['values'], colLabels=inday_test['columns'], fontsize=30, loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(6)
        plt.title("Test")
        
        plt.suptitle("Intraday Trading")
        pdf_inday.savefig()

        plt.figure(figsize=(12, 5))
        plt.axis('off')
        t = plt.table(cellText=realistic_test['values'], colLabels=realistic_test['columns'], fontsize=30, loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(6)
        plt.title("Test")
        plt.suptitle("Realistic Trading")
        pdf_inday.savefig()

        pdf_inday.close()

    def plot_results(self, ensemble_type):
        if ensemble_type == "ensemble":
            self.plot_ensemble_results()
        elif ensemble_type == "moe":
            self.plot_moe_results()
        else:
            raise ValueError("Invalid ensemble type")