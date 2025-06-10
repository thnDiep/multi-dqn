import torch
import torch.nn as nn
import torch.nn.functional as F

class MoeFlatNetwork(nn.Module):
    def __init__(self, num_experts=100, num_actions=3, hidden_dim=128, model_type='action'):
        super().__init__()
        self.num_experts = num_experts
        self.num_actions = num_actions
        self.model_type = model_type

        input_dim = num_experts * num_actions

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, expert_input):
        batch_size = expert_input.size(0)
        
        if self.model_type == 'action':
            # expert_input: (batch_size, num_experts)
            expert_one_hot = F.one_hot(expert_input, num_classes=self.num_actions).float()
            router_input = expert_one_hot.view(batch_size, -1)  # (batch_size, 100*3)
            expert_values = expert_one_hot
        else:
            router_input = expert_input.view(batch_size, -1)
            expert_values = expert_input

        router_logits = self.router(router_input)  # (batch_size, num_experts)
        expert_weights = F.softmax(router_logits, dim=-1)

        weights = expert_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
        output = torch.sum(weights * expert_values, dim=1)  # (batch_size, num_actions)
        
        # if self.model_type == 'q_value':
        #     output = F.softmax(output, dim=-1)

        return output, expert_weights

    def get_action(self, output):
        return torch.argmax(output, dim=-1)

class Moe2DNetwork(nn.Module):
    def __init__(self, num_experts=100, num_actions=3, hidden_dim=128, model_type='action', class_weights=None):
        super().__init__()
        self.num_experts = num_experts
        self.num_actions = num_actions
        self.model_type = model_type
        self.class_weights = class_weights

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

        # Router: from context → expert weights
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, expert_input, target_action=None, lambda_entropy=0.01):
        if self.model_type == 'action':
            # One-hot encode expert actions: (B, 100, 3)
            expert_input = F.one_hot(expert_input, num_classes=self.num_actions).float()
        
        # Encode each one-hot vector
        x = self.expert_encoder(expert_input) # (B, 100, hidden_dim)
        x = x + self.positional_embedding     # (B, 100, hidden_dim)
        x = self.transformer(x)               # (B, 100, hidden_dim)
        pooled = x.mean(dim=1)                # (B, hidden_dim)
        
        router_logits = self.router_mlp(pooled) # (B, 100)
        expert_weights = F.softmax(router_logits, dim=-1) # (B, 100)
        weights = expert_weights.unsqueeze(-1) # (B, 100, 1)

        # Combine expert outputs weighted by attention
        output = torch.sum(weights * expert_input, dim=1)
        return output, expert_weights
    
    def get_action(self, output):
        return torch.argmax(output, dim=-1)