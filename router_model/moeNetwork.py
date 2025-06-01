import torch
import torch.nn as nn
import torch.nn.functional as F

class MoeNetwork(nn.Module):
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

    def forward(self, expert_input, target_action=None):
        batch_size = expert_input.size(0)
        
        if self.model_type == 'action':
            # expert_input: (batch_size, num_experts)
            expert_one_hot = F.one_hot(expert_input, num_classes=self.num_actions).float()
            router_input = expert_one_hot.view(batch_size, -1)  # (batch_size, 100*3)
            expert_values = expert_one_hot
        else:  # q_value
            # expert_input: (batch_size, num_experts, num_actions)
            router_input = expert_input.view(batch_size, -1)  # (batch_size, 100*3)
            expert_values = expert_input

        router_logits = self.router(router_input)  # (batch_size, num_experts)
        expert_weights = F.softmax(router_logits, dim=-1)

        weights = expert_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
        final_distribution = torch.sum(weights * expert_values, dim=1)  # (batch_size, num_actions)
        
        if self.model_type == 'q_value':
            final_distribution = F.softmax(final_distribution, dim=-1)
            
        final_action = torch.argmax(final_distribution, dim=-1)

        loss = None
        if target_action is not None:
            loss = F.cross_entropy(final_distribution, target_action)

        return final_action, final_distribution, loss
