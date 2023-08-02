#%%
from torch.nn import BCEWithLogitsLoss
import torch
#%%loss stuff
input = torch.randn(3,12, requires_grad=True)
target = torch.empty(3,12).random_(2)
loss_fn = BCEWithLogitsLoss(reduction="none")

mask = torch.empty_like(target).random_(2).bool()
#%%
n_not_nan = mask.sum(1)
weights = torch.ones_like(input) / n_not_nan.unsqueeze(1)
#weights = weights * input.shape[1]
masked_weights = weights[mask]
masked_weights

loss = (loss_fn(input[mask],target[mask]) * masked_weights).sum()
loss / input.shape[0]

# %% matrix stuff
n_nodes = 50
n_graphs = 20
hidden_dim = 80
input_nodes = torch.randn(n_graphs, n_nodes+1, hidden_dim)
output_nodes = torch.randn(n_graphs, n_nodes+1, hidden_dim)

edge_type = input_nodes.view(n_graphs, n_nodes + 1, 1, -1) + output_nodes.view(n_graphs, 1, n_nodes + 1, -1)

# %%
