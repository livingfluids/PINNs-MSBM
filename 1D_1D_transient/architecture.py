import torch
import torch.nn as nn
import torch.nn.init as init
from dataclasses import dataclass
import config

# Fourier Input Transformation Layer (Tancik et al.) 
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, neurons, scale):
        super().__init__()  # inherit
        self.B = nn.Parameter(torch.randn(in_dim, neurons) * scale)

    def forward(self, x): 
        proj = x @ self.B  # can't do 'x @= self.B' because it breaks gradients
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

# Simple PINN Architecture
def buildSimplePINN(neurons):
    PINN = nn.Sequential(

        # 1 Fourier Input Layer
        FourierFeatures(in_dim=2, neurons=neurons, scale=config.SCALE),
        
        # 3 Hidden Layers
        nn.Linear(in_features=neurons * 2, out_features=neurons),
        config.ACTIVATION,
        # -
        nn.Linear(in_features=neurons, out_features=neurons),
        config.ACTIVATION,
        # - 
        nn.Linear(in_features=neurons, out_features=neurons),
        config.ACTIVATION,
        # -
        nn.Linear(in_features=neurons, out_features=neurons),
        config.ACTIVATION,
        # - 
        nn.Linear(in_features=neurons, out_features=neurons),
        config.ACTIVATION,

        # 1 Linear Output Layer
        nn.Linear(in_features=neurons, out_features=2)
    )
    return PINN

# Trial Function Data Class 
@dataclass 
class Trials:
    u_trial: torch.Tensor
    ϕ_trial: torch.Tensor

# Trial Functions 
def buildTrials(params, PINN) -> Trials:

    # u Trial Function
    def uTrial(y, t):
        y_grid, t_grid = torch.meshgrid(y.squeeze(), t.squeeze(), indexing='ij')    # [y, t], [y, t]
        y_flat, t_flat = y_grid.reshape(-1, 1), t_grid.reshape(-1, 1)               # [y·t, 1], [y·t, 1]
        # - 
        input_flat = torch.cat([y_flat, t_flat], dim=1)                             # [y·t, 2]
        output_flat = torch.sigmoid(PINN(input_flat)[:, 0:1])                       # [y·t, 1]
        # sigmoid(y, t) -> u
        # - 
        output_grid = output_flat.reshape(y.shape[0], t.shape[0], 1)                # [y, t, 1]
        # -
        return output_grid, output_flat                                             # [y, t, 1], [y·t, 1]

    # ϕ Trial Function
    def ϕTrial(y, t):
        y_grid, t_grid = torch.meshgrid(y.squeeze(), t.squeeze(), indexing='ij')    # [y, t], [y, t]
        y_flat, t_flat = y_grid.reshape(-1, 1), t_grid.reshape(-1, 1)               # [y·t, 1], [y·t, 1]
        input_flat = torch.cat([y_flat, t_flat], dim=1)                             # [y·t, 2]
        output_flat = params.ϕ_max * torch.sigmoid(PINN(input_flat)[:,1:2])         # [y·t, 1]
        # ϕ_max * sigmoid(y, t) -> ϕ
        # - 
        output_grid = output_flat.reshape(y.shape[0], t.shape[0], 1)                # [y, t, 1]
        # - 
        return output_grid, output_flat                                             # [y, t, 1], [y·t, 1]

    return Trials(u_trial=uTrial, ϕ_trial=ϕTrial)
