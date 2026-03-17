import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Tuple
import config

# Fourier Input Transformation Layer (Tancik et al.) 
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, neurons, scale):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, neurons) * scale)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

# Simple PINN Architecture
def buildSimplePINN(neurons):
    return nn.Sequential(

        # 1 Fourier Input Layer
        FourierFeatures(in_dim=2, neurons=neurons, scale=config.SCALE),

        # 5 Hidden Layers
        nn.Linear(neurons * 2, neurons),
        config.ACTIVATION,
        # - 
        nn.Linear(neurons, neurons),
        config.ACTIVATION,
        # - 
        nn.Linear(neurons, neurons),
        config.ACTIVATION,
        # - 
        nn.Linear(neurons, neurons),
        config.ACTIVATION,
        # - 
        nn.Linear(neurons, neurons),
        config.ACTIVATION,

        # 1 Linear Output Layer
        nn.Linear(neurons, 4),  # u, v, p, phi_raw
    )

# Trial Function Data Class
@dataclass
class Trials:
    u_trial: Callable[[torch.Tensor], torch.Tensor]
    v_trial: Callable[[torch.Tensor], torch.Tensor]
    p_trial: Callable[[torch.Tensor], torch.Tensor]
    ϕ_trial: Callable[[torch.Tensor], torch.Tensor]
    # uvpϕ_trial: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

# Build Trial Functions
def buildTrials(params, device, PINN) -> Trials:
    u_trial = lambda xy: torch.sigmoid(PINN(xy)[:, 0:1]) * 2 - 1
    v_trial = lambda xy: torch.sigmoid(PINN(xy)[:, 1:2]) * 2 - 1
    p_trial = lambda xy: PINN(xy)[:, 2:3]
    ϕ_trial = lambda xy: params.ϕ_max * torch.sigmoid(PINN(xy)[:, 3:4])
    # - 
    return Trials(
        u_trial=u_trial, 
        v_trial=v_trial, 
        p_trial=p_trial, 
        ϕ_trial=ϕ_trial)
