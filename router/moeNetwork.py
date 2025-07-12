import torch
import torch.nn as nn
import torch.nn.functional as F

class MoERouter(nn.Module):
    """
    Mô hình Router sử dụng Attention để chọn expert phù hợp nhất.
    Đầu vào gồm 4 phần: context (bối cảnh), reward (thưởng), risk (rủi ro), q_values (giá trị Q)
    Kết hợp cả 4 để tính điểm cho từng expert, sau đó chọn ra top-k expert.
    """
    def __init__(self, num_experts, hidden_dim=64, top_k=5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Mỗi encoder ánh xạ đầu vào sang không gian ẩn
        self.context_encoder = nn.Linear(68, hidden_dim)
        self.expert_encoder = nn.Linear(7, hidden_dim)
        self.router_scorer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_context, x_q_values, x_reward, x_risk):
        """
        Trả về:
        - action_logits: logits cho 3 actions (để dùng cho loss và prediction)
        - gate_weights: phân phối xác suất lên top-k expert (đã softmax)
        - expert_logits: điểm gốc của expert chưa chuẩn hóa (để dùng cho regularization loss)
        """
        context_encoded = self.context_encoder(x_context)
        context_expanded = context_encoded.unsqueeze(1).expand(-1, self.num_experts, -1)

        expert_inputs = torch.cat([x_q_values, x_reward, x_risk], dim=-1)  # (B,100,7)
        expert_encoded = self.expert_encoder(expert_inputs)

        combined = torch.cat([expert_encoded, context_expanded], dim=-1)  # (B,100,hidden*2)
        expert_logits = self.router_scorer(combined).squeeze(-1)  # (B,100)

        topk_val, topk_idx = torch.topk(expert_logits, self.top_k, dim=-1)
        sparse_weights = torch.full_like(expert_logits, float('-inf')).scatter(-1, topk_idx, topk_val)
        gate_weights = F.softmax(sparse_weights, dim=-1)  # (B,100)

        final_q_values = torch.bmm(gate_weights.unsqueeze(1), x_q_values).squeeze(1)  # (B, 3)
        action_logits = final_q_values

        return action_logits, gate_weights, expert_logits

    def get_action(self, action_logits):
        return torch.argmax(action_logits, dim=-1)

# class MoeFlatNetwork(nn.Module):
#     def __init__(self, num_experts=100, num_actions=3, hidden_dim=128):
#         super().__init__()
#         self.num_experts = num_experts
#         self.num_actions = num_actions

#         input_dim = num_experts * num_actions

#         self.router = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_experts),
#         )

#     def forward(self, expert_input):
#         batch_size = expert_input.size(0)

#         router_logits = self.router(expert_input.view(batch_size, -1))  # (batch_size, num_experts)
#         expert_weights = F.softmax(router_logits, dim=-1)

#         weights = expert_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
#         output = torch.sum(weights * expert_input, dim=1)  # (batch_size, num_actions)

#         return output, expert_weights

#     def get_action(self, output):
#         return torch.argmax(output, dim=-1)

# class Moe2DNetwork(nn.Module):
#     def __init__(self, num_experts=100, num_actions=3, hidden_dim=128):
#         super().__init__()
#         self.num_experts = num_experts
#         self.num_actions = num_actions

#         self.expert_encoder = nn.Sequential(
#             nn.Linear(num_actions, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )

#         # Learnable positional embeddings for experts
#         self.positional_embedding = nn.Parameter(torch.randn(1, num_experts, hidden_dim))

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
#         self.norm = nn.LayerNorm(hidden_dim)

#         # Router: from context → expert weights
#         self.router_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_experts)
#         )

#     def forward(self, expert_input):
#         x = self.expert_encoder(expert_input) # (B, 100, hidden_dim)
#         x = self.transformer(x + self.positional_embedding)               # (B, 100, hidden_dim)
#         x = self.norm(x)
#         pooled = x.mean(dim=1)                # (B, hidden_dim)

#         router_logits = self.router_mlp(pooled) # (B, 100)
#         expert_weights = F.softmax(router_logits, dim=-1) # (B, 100)
#         weights = expert_weights.unsqueeze(-1) # (B, 100, 1)

#         # Combine expert outputs weighted by attention
#         output = torch.sum(weights * expert_input, dim=1)
#         return output, expert_weights

#     def get_action(self, output):
#         return torch.argmax(output, dim=-1)

