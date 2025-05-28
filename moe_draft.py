# class MoEEnsemble:
#     def __init__(self, num_experts):
#         self.num_experts = num_experts
#         # Khởi tạo trọng số bằng nhau cho tất cả experts
#         self.weights = np.ones(num_experts) / num_experts
#         # Lưu lịch sử hiệu suất của từng expert
#         self.performance_history = {i: [] for i in range(num_experts)}
        
#     def update_weights(self, expert_performances):
#         """
#         Cập nhật trọng số dựa trên hiệu suất của các experts
#         expert_performances: dict chứa hiệu suất của từng expert
#         """
#         # Cập nhật lịch sử hiệu suất
#         for expert_id, perf in expert_performances.items():
#             self.performance_history[expert_id].append(perf)
            
#         # Tính trọng số mới dựa trên hiệu suất trung bình
#         avg_performances = {i: np.mean(perfs) for i, perfs in self.performance_history.items()}
#         total_perf = sum(avg_performances.values())
        
#         if total_perf > 0:
#             self.weights = np.array([avg_performances[i]/total_perf for i in range(self.num_experts)])
#         else:
#             self.weights = np.ones(self.num_experts) / self.num_experts
            
#     def predict(self, expert_predictions):
#         """
#         Kết hợp các dự đoán của experts dựa trên trọng số
#         expert_predictions: DataFrame chứa dự đoán của từng expert
#         """
#         weighted_predictions = np.zeros(len(expert_predictions))
        
#         for i in range(self.num_experts):
#             expert_pred = expert_predictions[f'iteration{i}'].values
#             weighted_predictions += self.weights[i] * expert_pred
            
#         # Chuyển đổi thành quyết định cuối cùng
#         final_predictions = np.zeros_like(weighted_predictions)
#         final_predictions[weighted_predictions > 0.5] = 1  # Mua
#         final_predictions[weighted_predictions < -0.5] = 2  # Bán
        
#         return pd.DataFrame(final_predictions, index=expert_predictions.index, columns=['ensemble'])

# def evaluate_moe_ensemble(num_walks, market, ensemble_folder, result_file):
#     """Đánh giá hiệu suất của MoE ensemble"""
#     # Khởi tạo MoE
#     moe = MoEEnsemble(num_walks)
    
#     # Đọc dữ liệu thị trường
#     market_data = pd.read_csv(f"./datasets/{market}Day.csv", index_col='Date')
    
#     # Đọc và xử lý từng walk
#     for j in range(num_walks):
#         df = pd.read_csv(f"{ensemble_folder}/walk{j}ensemble_test.csv", index_col='Date')
        
#         # Tính hiệu suất của expert
#         performance = calculate_expert_performance(df, market_data)
#         moe.update_weights({j: performance})
    
#     # Tạo báo cáo kết quả
#     pdf = PdfPages(f"{result_file}_moe.pdf")
#     plt.figure(figsize=(10, 5))
    
#     # Vẽ biểu đồ trọng số của các experts
#     plt.subplot(1, 2, 1)
#     plt.bar(range(num_walks), moe.weights)
#     plt.title('Expert Weights')
#     plt.xlabel('Expert ID')
#     plt.ylabel('Weight')
    
#     # Vẽ biểu đồ hiệu suất của các experts
#     plt.subplot(1, 2, 2)
#     for i in range(num_walks):
#         plt.plot(moe.performance_history[i], label=f'Expert {i}')
#     plt.title('Expert Performance History')
#     plt.xlabel('Time')
#     plt.ylabel('Performance')
#     plt.legend()
    
#     pdf.savefig()
#     pdf.close()

# def calculate_expert_performance(expert_predictions, market_data):
#     """Tính hiệu suất của một expert"""
#     performance = 0
#     for date, pred in expert_predictions.iterrows():
#         if date in market_data.index:
#             if pred['ensemble'] == 1:  # Mua
#                 performance += (market_data.at[date,'Close'] - market_data.at[date,'Open'])/market_data.at[date,'Open']
#             elif pred['ensemble'] == 2:  # Bán
#                 performance += -(market_data.at[date,'Close'] - market_data.at[date,'Open'])/market_data.at[date,'Open']
#     return performance