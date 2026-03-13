import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.market_config import get_market_config
from evaluation.trading import IndayTrading

class Evaluation:
    def __init__(self, model_name, market, input_dir, result_dir, final_decision_dir, stop_loss_pct=0.02, take_profit_pct=0.04):
        self.model_name = model_name
        self.market = market

        self.num_walks = get_market_config(market)["num_walks"]

        self.input_dir = input_dir
        self.result_dir = result_dir
        self.final_decision_dir = final_decision_dir
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.final_action = "ensemble"

    def plot_ensemble_results(self):
        def evaluate(consensus_threshold=0, phase="test"):
            def full_ensemble(df):
                """Calculate ensemble with 100% consensus"""
                m1 = df.eq(1).all(axis=1)
                m2 = df.eq(2).all(axis=1)
                local_df = df.copy()
                local_df[self.final_action] = np.select([m1, m2], [1, 2], 0)
                local_df = local_df.drop(local_df.columns.difference([self.final_action]), axis=1)
                return local_df

            def perc_ensemble(df, thr=0.7):
                """Calculate ensemble with specific consensus threshold"""
                c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
                c2 = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
                return pd.DataFrame(np.select([c1, c2], [1, 2], 0), index=df.index, columns=[self.final_action])

            inday_trading = IndayTrading(self.market)

            # Process each walk file once
            for j in range(self.num_walks):
                df = pd.read_csv(f"{self.input_dir}/walk{j}ensemble_{phase}.csv", index_col='Date')
                df.index = pd.to_datetime(df.index)

                if consensus_threshold == 0:
                    df = full_ensemble(df)
                else:
                    df = perc_ensemble(df, consensus_threshold)

                df.to_csv(f"{self.final_decision_dir}/walk{j}_{phase}_{consensus_threshold}.csv")
                inday_trading.trading_for_each_walk(df, j)  # Evaluate inday trading

            return inday_trading.get_total_walk_result()

        pdf_inday_trading = PdfPages(f"{self.result_dir}/{self.model_name}.pdf")
        thresholds = [0, 0.9, 0.8, 0.7, 0.6]
        titles = ["FULL ENSEMBLE", "90% ENSEMBLE", "80% ENSEMBLE", "70% ENSEMBLE", "60% ENSEMBLE"]
        
        for threshold, title in zip(thresholds, titles):
            results = evaluate(threshold)

            plt.figure(figsize=(12, 5))
            plt.axis('off')
            t = plt.table(cellText=results['values'], colLabels=results['columns'], fontsize=30, loc='center')
            t.auto_set_font_size(False)
            t.set_fontsize(6)
            plt.title("Test")
            plt.suptitle(title)
            pdf_inday_trading.savefig()

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            for i, (equity_curve, dates) in enumerate(results['walk_data']):
                plt.plot(dates, equity_curve, linewidth=1, label=f'Walk {i}')

            plt.title('Equity Curves Across All Walks')
            plt.xlabel('Date')
            plt.ylabel('Account Balance')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.subplot(1, 2, 2)
            plt.plot(results['dates'], results['equity'], 'b-', linewidth=1, label='Cumulative Equity')
            plt.title('Cumulative Equity Curve Across All Walks')
            plt.xlabel('Date')
            plt.ylabel('Account Balance')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf_inday_trading.savefig()
        pdf_inday_trading.close()

    def plot_moe_results(self):
        def evaluate(phase="test"):
            inday_trading = IndayTrading(self.market)

            # Process each walk file once
            for j in range(self.num_walks):
                df = pd.read_csv(f"{self.input_dir}/walk{j}_{phase}.csv", index_col='Date')
                df.index = pd.to_datetime(df.index)
                inday_trading.trading_for_each_walk(df, j)

            result = inday_trading.get_total_walk_result()
            return result

        pdf_inday_trading = PdfPages(f"{self.result_dir}/moe_{self.model_name}.pdf")
        results = evaluate()

        plt.figure(figsize=(12, 5))
        plt.axis('off')
        t = plt.table(cellText=results['values'], colLabels=results['columns'], fontsize=30, loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(6)
        plt.title("Test")
        plt.suptitle("Intraday Trading")
        pdf_inday_trading.savefig()

        plt.subplot(1, 2, 1)
        for i, (equity_curve, dates) in enumerate(results['walk_data']):
            plt.plot(dates, equity_curve, linewidth=1, label=f'Walk {i}')

        plt.title('Equity Curves Across All Walks')
        plt.xlabel('Date')
        plt.ylabel('Account Balance')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.plot(results['dates'], results['equity'], 'b-', linewidth=1, label='Cumulative Equity')
        plt.title('Cumulative Equity Curve Across All Walks')
        plt.xlabel('Date')
        plt.ylabel('Account Balance')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf_inday_trading.savefig()
        pdf_inday_trading.close()

    def plot_results(self, ensemble_type=None):
        if ensemble_type == "ensemble":
            self.plot_ensemble_results()
        elif ensemble_type == "moe":
            self.plot_moe_results()
        else:
            raise ValueError("Invalid ensemble type")