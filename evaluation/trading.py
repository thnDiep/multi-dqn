import numpy as np
import pandas as pd

from utils.market_config import get_market_point_value


class IndayTrading:
    def __init__(self, market, initial_balance=10000):
        self.market = market
        self.market_point_value = get_market_point_value(market)

        self.market_data = pd.read_csv(f"./datasets/{market}Day.csv", index_col='Date')
        self.market_data.index = pd.to_datetime(self.market_data.index)

        self.initial_balance = initial_balance
        self.doll_sum = 0
        self.rew_sum = 0
        self.pos_sum = 0
        self.neg_sum = 0
        self.cov_sum = 0
        self.num_sum = 0
        self.values = []
        self.daily_returns = []
        self.equity_curves = []
        self.dates  = []
        self.columns = ["Iteration", "Reward%", "#W", "#L", "W/L", "Dollars", "Coverage", "Accuracy", "Sortino", "MDD%", "Profit Factor"]
        self.final_action_column = "ensemble"

    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02):
        if not returns:
            return 0
        returns = np.array(returns)
        daily_rf = risk_free_rate / 252

        downside_diff = np.minimum(returns - daily_rf, 0)  # Only negative deviations
        downside_deviation = np.sqrt(np.mean(downside_diff ** 2))
        if downside_deviation == 0:
            return float('inf')
        excess_return = np.mean(returns) - daily_rf
        return np.sqrt(252) * (excess_return / downside_deviation)

    def calculate_mdd(self, equity_curve):
        if not equity_curve:
            return 0
        
        equity_curve = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        mdd = np.max(drawdown) * 100
        return mdd

    def calculate_profit_factor(self, gains, losses):
        if not losses or sum(losses) == 0:
            return float('inf')
        
        return sum(gains) / abs(sum(losses))

    def calculate_position_size(self, balance, price):
        return balance / price

    def trading_for_each_walk(self, df, current_walk):
        num = 0
        rew = 0
        pos = 0
        neg = 0
        doll = 0
        cov = 0

        balance = self.initial_balance
        daily_returns = []
        equity_curve = []
        dates = []

        gains = []
        losses = []
        
        for date, i in df.iterrows():
            num += 1
            open_price = self.market_data.at[date, 'Open']
            close_price = self.market_data.at[date, 'Close']
            action = i[self.final_action_column]

            if date in self.market_data.index:
                dates.append(date)

                if self.market_point_value:
                    position_size = self.market_point_value
                else:
                    position_size = self.calculate_position_size(balance, open_price)

                if action == 1:  # Long
                    pct_return = (close_price - open_price) / open_price
                    rew += pct_return
                    pos += 1 if pct_return > 0 else 0
                    neg += 1 if pct_return < 0 else 0
                    abs_return = (close_price - open_price) * position_size
                    doll += abs_return
                    balance += abs_return
                    cov += 1
                    daily_returns.append(pct_return)
                    if pct_return > 0:
                        gains.append(abs_return)
                    else:
                        losses.append(abs_return)
                        
                elif action == 2:  # Short
                    pct_return = -(close_price - open_price) / open_price
                    rew += pct_return
                    pos += 1 if pct_return > 0 else 0
                    neg += 1 if pct_return < 0 else 0
                    abs_return = -(close_price - open_price) * position_size
                    doll += abs_return
                    balance += abs_return
                    cov += 1
                    daily_returns.append(pct_return)
                    if pct_return > 0:
                        gains.append(abs_return)
                    else:
                        losses.append(abs_return)

                equity_curve.append(balance)
        acc = pos / cov if cov > 0 else 0
        cov_rate = cov / num if num > 0 else 0
        sharpe = self.calculate_sortino_ratio(daily_returns)
        mdd_pct = self.calculate_mdd(equity_curve)
        profit_factor = self.calculate_profit_factor(gains, losses)

        self.values.append([
            str(round(current_walk,2)),
            str(round(rew * 100, 2)),
            str(round(pos,2)),
            str(round(neg,2)),
            str(round(pos/neg,2)) if neg > 0 else float('inf'),
            str(round(doll,2)),
            str(round(cov_rate,2)),
            str(round(acc,2)),
            str(round(sharpe,2)),
            str(round(mdd_pct,2)),
            str(round(profit_factor,2)) if profit_factor != float('inf') else float('inf'),
        ])
        
        self.rew_sum += rew
        self.pos_sum += pos
        self.neg_sum += neg
        self.doll_sum += doll
        self.cov_sum += cov
        self.num_sum += num
        self.daily_returns.extend(daily_returns)

        self.equity_curves.append(equity_curve)
        self.dates.append(dates)

    def get_total_walk_result(self):
        walk_data = list(zip(self.equity_curves, self.dates))

        date, cumulative_balance = self.get_cumulative_equity(walk_data)
        total_sortino = self.calculate_sortino_ratio(self.daily_returns)
        total_mdd_pct = self.calculate_mdd(cumulative_balance)

        total_profit_factor = self.calculate_profit_factor(
            [r for r in self.daily_returns if r > 0],
            [r for r in self.daily_returns if r < 0]
        )

        self.values.append([
            "sum",
            str(round(self.rew_sum * 100,2)),
            str(round(self.pos_sum,2)),
            str(round(self.neg_sum,2)),
            str(round(self.pos_sum/self.neg_sum,2)) if self.neg_sum > 0 else float('inf'),
            str(round(self.doll_sum,2)),
            str(round(self.cov_sum/self.num_sum,2) if self.num_sum > 0 else 0),
            str(round(self.pos_sum/self.cov_sum,2) if self.cov_sum > 0 else 0),
            str(round(total_sortino,2)),
            str(round(total_mdd_pct,2)),
            str(round(total_profit_factor,2)) if total_profit_factor != float('inf') else float('inf'),
        ])

        return {
            'values': self.values,
            'columns': self.columns,
            'walk_data': walk_data,
            'dates': date,
            'equity': cumulative_balance
        }

    def get_cumulative_equity(self, walk_data):
        all_dates = []
        cumulative_balance = []
        current_balance = self.initial_balance

        for equity_curve, dates in walk_data:
            if dates:
                all_dates.append(dates[0])
                cumulative_balance.append(current_balance)
            for i in range(1, len(equity_curve)):
                daily_pnl = equity_curve[i] - equity_curve[i - 1]
                current_balance += daily_pnl
                all_dates.append(dates[i - 1])
                cumulative_balance.append(current_balance)

        return all_dates, cumulative_balance
