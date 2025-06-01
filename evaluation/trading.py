import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class IndayTrading:
    def __init__(self, market_data):
        self.market_data = market_data
        self.doll_sum = 0
        self.rew_sum = 0
        self.pos_sum = 0
        self.neg_sum = 0
        self.cov_sum = 0
        self.num_sum = 0
        self.values = []
        self.columns = ["Iteration", "Reward%", "#Wins", "#Losses", "Dollars", "Coverage", "Accuracy"]
        self.final_action_column = "ensemble"

    def trading_for_each_walk(self, df, current_walk):
        num = 0
        rew = 0
        pos = 0
        neg = 0
        doll = 0
        cov = 0
        
        for date, i in df.iterrows():
            num += 1
            if date in self.market_data.index:
                if (i[self.final_action_column] == 1):  # Long
                    rew += (self.market_data.at[date,'Close'] - self.market_data.at[date,'Open']) / self.market_data.at[date,'Open']
                    pos += 1 if rew > 0 else 0
                    neg += 0 if rew > 0 else 1
                    doll += (self.market_data.at[date,'Close'] - self.market_data.at[date,'Open']) * 50
                    cov += 1
                elif (i[self.final_action_column] == 2):  # Short
                    rew += -(self.market_data.at[date,'Close'] - self.market_data.at[date,'Open']) / self.market_data.at[date,'Open']
                    neg += 0 if -rew > 0 else 1
                    pos += 1 if -rew > 0 else 0
                    doll += -(self.market_data.at[date,'Close'] - self.market_data.at[date,'Open']) * 50
                    cov += 1
        
        self.values.append([
            str(round(current_walk,2)),
            str(round(rew,2)),
            str(round(pos,2)),
            str(round(neg,2)),
            str(round(doll,2)),
            str(round(cov/num,2)),
            (str(round(pos/cov,2)) if (cov>0) else "0")
        ])
        
        self.doll_sum += doll
        self.rew_sum += rew
        self.pos_sum += pos
        self.neg_sum += neg
        self.cov_sum += cov
        self.num_sum += num


    def get_total_walk_result(self):
        self.values.append([
            "sum",
            str(round(self.rew_sum,2)),
            str(round(self.pos_sum,2)),
            str(round(self.neg_sum,2)),
            str(round(self.doll_sum,2)),
            str(round(self.cov_sum/self.num_sum,2)),
            (str(round(self.pos_sum/self.cov_sum,2)) if (self.cov_sum>0) else "0")
        ])
        return self.values, self.columns


class Position:
    def __init__(self, entry_price, entry_date, position_type, stop_loss, take_profit, quantity):
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.position_type = position_type  # 1: long, 2: short
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.quantity = quantity  # Số lượng cổ phiếu
        self.exit_price = None
        self.exit_date = None
        self.pnl = 0
        self.exit_reason = None  # 'stop_loss', 'take_profit', 'signal'

    
class RealisticTrading:
    def __init__(self, market_data, stop_loss_pct=0.02, take_profit_pct=0.04, initial_balance=10000, commission_rate=0.001, use_sl_tp=True):
        self.initial_balance = initial_balance
        self.market_data = market_data
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.commission_rate = commission_rate  # Phí giao dịch 0.1%
        self.use_sl_tp = use_sl_tp  # Thêm tham số để kiểm soát việc sử dụng stop loss và take profit
        self.walk_pnls = []  # Lưu lợi nhuận/thua lỗ của từng walk
        self.total_trades = 0
        self.total_return_pct = 0
        self.total_win_loss_rate = 0
        self.total_profit_factor = 0
        self.total_avg_win = 0
        self.total_avg_loss = 0
        self.max_drawdown = 0
        self.total_sharpe_ratio = 0
        self.walk_count = 0
        self.positions = []  # Lưu trữ tất cả các vị thế

        self.final_action_column = "ensemble"

        self.values = []
        self.columns = ["Iteration", "Final Balance", "Total Trades", "Return %", "W/L Rate", "Profit Factor", "Avg Win", "Avg Loss", "Max Drawdown", "Sharpe Ratio"]

        self.equity_curves = []  # Lưu trữ tất cả các equity curves
        self.dates = []  # Lưu trữ các ngày tương ứng

    def calculate_metrics(self, positions, equity_curve, daily_returns):
        total_trades = len(positions)
        final_balance = equity_curve[-1]
        return_pct = (final_balance - equity_curve[0]) / equity_curve[0]

        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0

        for position in positions:
            if position.pnl > 0:
                winning_trades += 1
                total_profit += position.pnl
            elif position.pnl < 0:
                losing_trades += 1
                total_loss += abs(position.pnl)

        win_loss_rate = winning_trades / losing_trades if losing_trades > 0 else float('inf')
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0

        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_drawdown = np.max(drawdown)

        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        return final_balance, total_trades, return_pct, win_loss_rate, profit_factor, avg_win, avg_loss, max_drawdown, sharpe_ratio   

    def trading_for_each_walk(self, df, current_walk):
        """Thực hiện giao dịch cho một walk"""
        positions = []
        daily_returns = []
        equity_curve = []
        current_balance = self.initial_balance
        active_position = None
        dates = []
        
        for date, row in df.iterrows():
            if date not in self.market_data.index:
                continue

            dates.append(date)
            current_price = self.market_data.at[date, 'Close']

            # Kiểm tra điều kiện đóng vị thế
            signal_type = None
            if active_position is not None:
                # Đóng vị thế nếu có tín hiệu đảo chiều
                if (active_position.position_type == 1 and row[self.final_action_column] == 2) or \
                   (active_position.position_type == 2 and row[self.final_action_column] == 1):
                    signal_type = 'signal'

                elif self.use_sl_tp:  # Chỉ kiểm tra stop loss và take profit nếu use_sl_tp=True
                    # Nếu không có tín hiệu đảo chiều thì kiểm tra stop_loss và take_profit
                    is_stop_loss = (active_position.position_type == 1 and current_price <= active_position.stop_loss) or \
                                 (active_position.position_type == 2 and current_price >= active_position.stop_loss)
                    is_take_profit = (active_position.position_type == 1 and current_price >= active_position.take_profit) or \
                                   (active_position.position_type == 2 and current_price <= active_position.take_profit)
                    if is_stop_loss:
                        signal_type = 'stop_loss'
                    elif is_take_profit:
                        signal_type = 'take_profit'

            if active_position is not None and signal_type is not None:
                # Sử dụng giá đóng cửa để đóng vị thế
                active_position.exit_price = current_price
                active_position.exit_date = date
                active_position.exit_reason = signal_type
                # Tính toán lợi nhuận/thua lỗ bao gồm phí giao dịch
                if active_position.position_type == 1:  # Long
                    pnl = (current_price - active_position.entry_price) * active_position.quantity
                else:  # Short
                    pnl = (active_position.entry_price - current_price) * active_position.quantity
                
                commission = (active_position.entry_price + current_price) * active_position.quantity * self.commission_rate
                pnl -= commission
                current_balance += pnl
                active_position.pnl = pnl
                positions.append(active_position)
                self.positions.append(active_position)
                active_position = None

            # Mở vị thế mới
            if active_position is None and row[self.final_action_column] != 0 and signal_type != 'signal':
                quantity = current_balance / current_price
                if row[self.final_action_column] == 1:  # Long
                    stop_loss = current_price * (1 - self.stop_loss_pct) if self.use_sl_tp else None
                    take_profit = current_price * (1 + self.take_profit_pct) if self.use_sl_tp else None
                    active_position = Position(current_price, date, 1, stop_loss, take_profit, quantity)
                elif row[self.final_action_column] == 2:  # Short
                    stop_loss = current_price * (1 + self.stop_loss_pct) if self.use_sl_tp else None
                    take_profit = current_price * (1 - self.take_profit_pct) if self.use_sl_tp else None
                    active_position = Position(current_price, date, 2, stop_loss, take_profit, quantity)

            # Cập nhật đường equity
            equity_curve.append(current_balance)
            if len(positions) > 0 and positions[-1].exit_date == date:
                daily_return = positions[-1].pnl / equity_curve[-2]
                daily_returns.append(daily_return)
            else:
                daily_returns.append(0)

        # Tính metrics và thêm vào values
        final_balance, total_trades, return_pct, win_loss_rate, profit_factor, avg_win, avg_loss, max_drawdown, sharpe_ratio = self.calculate_metrics(positions, equity_curve, daily_returns)
        
        # Lưu lợi nhuận/thua lỗ của walk này
        walk_pnl = final_balance - self.initial_balance
        self.walk_pnls.append(walk_pnl)
        
        self.values.append([
            str(current_walk),
            round(final_balance, 2),
            total_trades,
            round(return_pct, 2),
            round(win_loss_rate, 2),
            round(profit_factor, 2),
            round(avg_win, 2),
            round(avg_loss, 2),
            round(max_drawdown, 2),
            round(sharpe_ratio, 2)
        ])
        
        self.total_trades += total_trades
        self.total_return_pct += return_pct
        self.total_win_loss_rate += win_loss_rate
        self.total_profit_factor += profit_factor
        self.total_avg_win += avg_win
        self.total_avg_loss += avg_loss
        self.max_drawdown = max(self.max_drawdown, max_drawdown)
        self.total_sharpe_ratio += sharpe_ratio
        self.walk_count += 1

        # Lưu equity curve và dates
        self.equity_curves.append(equity_curve)
        self.dates.append(dates)

    def get_total_walk_result(self):
        """Tính toán kết quả tổng hợp từ tất cả các walks"""
        if not self.values:
            return None
        
        # Tính final balance bằng cách cộng tất cả lời/lỗ vào initial balance
        total_pnl = sum(self.walk_pnls)
        final_balance = self.initial_balance + total_pnl
        
        # Thêm dòng tổng
        self.values.append([
            "Sum",
            round(final_balance, 2),  # Sử dụng tổng lời/lỗ từ tất cả walks
            self.total_trades,
            round(self.total_return_pct / self.walk_count, 2),
            round(self.total_win_loss_rate / self.walk_count, 2),
            round(self.total_profit_factor / self.walk_count, 2),
            round(self.total_avg_win / self.walk_count, 2),
            round(self.total_avg_loss / self.walk_count, 2),
            round(self.max_drawdown, 2),
            round(self.total_sharpe_ratio / self.walk_count, 2)
        ])
        
        return self.values, self.columns

    def plot_equity_curve(self, output_file):
        """Vẽ biểu đồ Equity Curve cho tất cả các walks"""
        pdf = PdfPages(output_file)
            
        plt.figure(figsize=(15, 8))

        walk_data = list(zip(self.equity_curves, self.dates))
        for i, (equity_curve, dates) in enumerate(walk_data):
            plt.plot(dates, equity_curve, linewidth=1, label=f'Walk {i}')

        plt.title('Equity Curves Across All Walks')
        plt.xlabel('Date')
        plt.ylabel('Account Balance')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Vẽ biểu đồ equity curve tổng hợp
        plt.figure(figsize=(15, 8))
        all_dates = []
        cumulative_balance = []
        current_balance = self.initial_balance

        for equity_curve, dates in walk_data:
            # Bắt đầu mỗi walk từ ngày đầu tiên của walk đó
            if dates:
                all_dates.append(dates[0])
                cumulative_balance.append(current_balance)
            for i in range(1, len(equity_curve)):
                daily_pnl = equity_curve[i] - equity_curve[i-1]
                current_balance += daily_pnl
                all_dates.append(dates[i-1])
                cumulative_balance.append(current_balance)
        plt.plot(all_dates, cumulative_balance, 'b-', linewidth=1, label='Cumulative Equity')
        plt.title('Cumulative Equity Curve Across All Walks')
        plt.xlabel('Date')
        plt.ylabel('Account Balance')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Vẽ biểu đồ với các điểm vào lệnh và thoát lệnh cho từng walk
        for walk_idx, (equity_curve, dates) in enumerate(walk_data):
            plt.figure(figsize=(15, 8))
            # Vẽ đường equity curve của walk
            # plt.plot(dates, equity_curve, 'b-', linewidth=1, label='Equity Curve')
            # Vẽ đường chuyển động giá Close
            close_prices = [self.market_data.at[date, 'Close'] for date in dates if date in self.market_data.index]
            if len(close_prices) == len(dates):
                plt.plot(dates, close_prices, 'orange', linewidth=1, label='Close Price')
            # Lấy các vị thế của walk hiện tại
            walk_positions = [pos for pos in self.positions if pos.entry_date in dates]
            entry_label_flags = {'Long Entry': False, 'Short Entry': False}
            exit_label_flags = {'Stop Loss': False, 'Take Profit': False, 'Signal Exit': False}
            for i, position in enumerate(walk_positions):
                entry_idx = dates.index(position.entry_date)
                if entry_idx < len(close_prices):
                    entry_balance = close_prices[entry_idx]
                    color = 'g' if position.position_type == 1 else 'r'
                    label = 'Long Entry' if position.position_type == 1 else 'Short Entry'
                    plot_label = label if not entry_label_flags[label] else ""
                    entry_label_flags[label] = True
                    plt.scatter(position.entry_date, entry_balance, 
                                color=color, marker='^' if position.position_type == 1 else 'v', 
                                s=100, label=plot_label)
                if position.exit_date:
                    exit_idx = dates.index(position.exit_date)
                    if exit_idx < len(close_prices):
                        exit_balance = close_prices[exit_idx]
                        if position.exit_reason == 'stop_loss':
                            color = 'r'
                            marker = 'o'
                            label = 'Stop Loss'
                        elif position.exit_reason == 'take_profit':
                            color = 'g'
                            marker = 'o'
                            label = 'Take Profit'
                        else:
                            color = 'b'
                            marker = 'o'
                            label = 'Signal Exit'
                        plot_label = label if not exit_label_flags[label] else ""
                        exit_label_flags[label] = True
                        plt.scatter(position.exit_date, exit_balance, 
                                    color=color, marker=marker, s=100, 
                                    label=plot_label)
            plt.title(f'Trading Points for Walk {walk_idx}')
            plt.xlabel('Date')
            plt.ylabel('Account Balance')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        pdf.close()