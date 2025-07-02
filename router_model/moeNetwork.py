import torch
import torch.nn as nn
import torch.nn.functional as F

class MoeFlatNetwork(nn.Module):
    def __init__(self, num_experts=100, num_actions=3, hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.num_actions = num_actions

        input_dim = num_experts * num_actions

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, expert_input):
        batch_size = expert_input.size(0)
        
        router_logits = self.router(expert_input.view(batch_size, -1))  # (batch_size, num_experts)
        expert_weights = F.softmax(router_logits, dim=-1)

        weights = expert_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
        output = torch.sum(weights * expert_input, dim=1)  # (batch_size, num_actions)
        
        return output, expert_weights

    def get_action(self, output):
        return torch.argmax(output, dim=-1)

class Moe2DNetwork(nn.Module):
    def __init__(self, num_experts=100, num_actions=3, hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.num_actions = num_actions

        self.expert_encoder = nn.Sequential(
            nn.Linear(num_actions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Learnable positional embeddings for experts
        self.positional_embedding = nn.Parameter(torch.randn(1, num_experts, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden_dim)

        # Router: from context → expert weights
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, expert_input):
        x = self.expert_encoder(expert_input) # (B, 100, hidden_dim)
        x = self.transformer(x + self.positional_embedding)               # (B, 100, hidden_dim)
        x = self.norm(x)
        pooled = x.mean(dim=1)                # (B, hidden_dim)
        
        router_logits = self.router_mlp(pooled) # (B, 100)
        expert_weights = F.softmax(router_logits, dim=-1) # (B, 100)
        weights = expert_weights.unsqueeze(-1) # (B, 100, 1)

        # Combine expert outputs weighted by attention
        output = torch.sum(weights * expert_input, dim=1)
        return output, expert_weights
    
    def get_action(self, output):
        return torch.argmax(output, dim=-1)