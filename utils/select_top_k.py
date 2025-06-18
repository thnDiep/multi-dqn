import numpy as np
import pandas as pd

def select_top_k_experts_by_group(df, k=30):
    """
    Giống hệt hàm select_top_k_diverse_experts nhưng nhóm theo expert (block 3 cột Q-values)
    """
    # Lấy tất cả cột Q-values
    qvalue_cols = [col for col in df.columns if col.startswith("iteration")]
    num_experts = len(qvalue_cols) // 3  # Không cố định là 100

    # Reshape về (N, num_experts, 3)
    qvalues = df[qvalue_cols].values.reshape(-1, num_experts, 3).astype(np.float32)

    # Tính std của mỗi expert theo (N, 3) → std tổng hợp trên cả 3 hành động
    expert_std = qvalues.std(axis=(0, 2))  # shape: (num_experts,)

    # Lấy chỉ số của top-k expert có std cao nhất
    topk_indices = np.argsort(expert_std)[-k:]

    # Lấy lại tên cột của các expert tương ứng
    selected_cols = []
    for i in topk_indices:
        for j in range(3):  # Buy, Hold, Sell
            selected_cols.append(f"iteration{i}_q{j}")

    # Giữ lại cột Date và label cuối
    meta_cols = ['Date', df.columns[-1]]
    new_df = df[meta_cols + selected_cols]
    return new_df