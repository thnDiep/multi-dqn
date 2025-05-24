import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from market_config import get_market_config
from trading import IndayTrading, RealisticTrading
import os

class Evaluation:
    def __init__(self, model_name, market, ensemble_dir, result_dir, final_decision_dir, stop_loss_pct=0.02, take_profit_pct=0.04):
        self.model_name = model_name
        self.market = market
        self.market_data = pd.read_csv(f"./datasets/{market}Day.csv", index_col='Date')
        self.market_data.index = pd.to_datetime(self.market_data.index)
        self.num_walks = get_market_config(market)["num_walks"]

        self.ensemble_dir = ensemble_dir
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
        local_df[self.final_action] = np.select([m1, m2], [1, -1], 0)
        local_df = local_df.drop(local_df.columns.difference([self.final_action]), axis=1)
        return local_df

    def perc_ensemble(self, df, thr=0.7):
        """Calculate ensemble with specific consensus threshold"""
        c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
        c2 = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
        return pd.DataFrame(np.select([c1, c2], [1, -1], 0), index=df.index, columns=[self.final_action])

    def evaluate(self, consensus_threshold=0, type="test"):
        """
        Evaluate model performance with both inday trading and realistic trading strategies
        """
        # Initialize results storage
        inday_trading = IndayTrading(self.market_data)
        # Open this comment to evaluate the results with sl and tp
        realistic_trading = RealisticTrading(self.market_data, self.stop_loss_pct, self.take_profit_pct)

        # Open this comment to evaluate the results without sl and tp
        # realistic_trading = RealisticTrading(self.market_data, self.stop_loss_pct, self.take_profit_pct, use_sl_tp=False)

        # Process each walk file once
        for j in range(self.num_walks):
            df = pd.read_csv(f"{self.ensemble_dir}/walk{j}ensemble_{type}.csv", index_col='Date')
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
        
        # Open this comment to plot the results with sl and tp
        realistic_trading.plot_equity_curve(f"{self.result_dir}/equity_curve_{consensus_threshold}.pdf")

        # Open this comment to plot the results without sl and tp
        # realistic_trading.plot_equity_curve(f"{self.result_dir}/no_sl_tp/equity_curve_{consensus_threshold}_no_sl_tp.pdf")

        return {
            'inday': {'values': inday_values, 'columns': inday_columns},
            'realistic': {'values': realistic_values, 'columns': realistic_columns}
        }
    
    def plot_ensemble_results(self):
        """Plot ensemble results tables with different thresholds"""
        # Open this comment to plot the results with sl and tp
        pdf_inday = PdfPages(f"{self.result_dir}/inday_trading.pdf")
        pdf_realistic = PdfPages(f"{self.result_dir}/realistic_trading.pdf")

        # Open this comment to plot the results without sl and tp
        # pdf_inday = PdfPages(f"{self.result_dir}/no_sl_tp/inday_trading_no_sl_tp.pdf")
        # pdf_realistic = PdfPages(f"{self.result_dir}/no_sl_tp/realistic_trading_no_sl_tp.pdf")
        
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

