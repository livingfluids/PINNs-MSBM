import torch
import torch.nn as nn
import torch.nn.init as init
from dataclasses import dataclass
import numpy as np
import config

# Fourier Input Transformation Layer (Tancik et al.) 
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, neurons, scale):
        super().__init__()  # inherit
        self.B = nn.Parameter(torch.randn(in_dim, neurons) * scale)

    def forward(self, x): 
        proj = x @ self.B  # cant do 'x @= self.B' because it breaks gradients?
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

# Simple PINN Architecture
def buildSimplePINN(neurons):
    PINN = nn.Sequential(

        # 1 Fourier Input Layer
        FourierFeatures(in_dim=1, neurons=neurons, scale=config.SCALE),
        
        # 5 Hidden Layers
        nn.Linear(in_features=neurons * 2, out_features=neurons),
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

# Build Trial Functions
def buildTrials(params, device, PINN) -> Trials:
    u_trial = lambda y: torch.sigmoid(PINN(torch.cat([y], dim=1))[:,0:1]) #* (1 + y.to(device)) * (1 - y.to(device))  # torch.Size([y, 1])
     # sigmoid(y) -> u
    ϕ_trial = lambda y: params.ϕ_max * torch.sigmoid(PINN(torch.cat([y], dim=1))[:,1:2]) # * (1 + y.to(device)) * (1 - y.to(device))  # torch.Size([y, 1])
    # ϕ_max * sigmoid(y) -> ϕ
    
    return Trials(u_trial=u_trial, ϕ_trial=ϕ_trial)
    