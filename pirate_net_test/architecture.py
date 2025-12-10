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

# PirateNet Gates (Wang et al.)
class Gates(nn.Module):
    def __init__(self, in_dim, neurons, scale):
        super().__init__()  # inherit
        self.W1 = nn.Parameter(torch.randn(2 * neurons, neurons, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(2 * neurons, neurons, requires_grad=True))
        self.b1 = nn.Parameter(torch.zeros(neurons, requires_grad=True))
        self.b2 = nn.Parameter(torch.zeros(neurons, requires_grad=True))
        self.σ  = nn.ReLU()
        self.Φ = FourierFeatures(in_dim=in_dim, neurons=neurons, scale=scale)

        # Xavier Initialization
        init.xavier_uniform_(self.W1)
        init.xavier_uniform_(self.W2)

    def forward(self, x):
        Φ = self.Φ(x)
        U = self.σ(Φ @ self.W1 + self.b1)
        V = self.σ(Φ @ self.W2 + self.b2)
        return U, V

# PirateNet Block (Wang et al.)
class PirateNetBlock(nn.Module):
    def __init__(self, in_dim, neurons, scale, α_init):
        super().__init__()  # inherit
        self.W1 = nn.Parameter(torch.randn(in_dim, neurons, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(neurons, neurons, requires_grad=True))
        self.W3 = nn.Parameter(torch.randn(neurons, neurons, requires_grad=True))
        self.b1 = nn.Parameter(torch.zeros(neurons, requires_grad=True))
        self.b2 = nn.Parameter(torch.zeros(neurons, requires_grad=True))
        self.b3 = nn.Parameter(torch.zeros(neurons, requires_grad=True))
        self.σ  = nn.ReLU()
        self.α  = nn.Parameter(torch.tensor([α_init]))
        self.G = Gates(in_dim=in_dim, neurons=neurons, scale=scale)
        
        # Xavier Initialization
        init.xavier_uniform_(self.W1)
        init.xavier_uniform_(self.W2)
        init.xavier_uniform_(self.W3)

    def forward(self, x): 
        U, V = self.G(x)
        f = self.σ(x @ self.W1 + self.b1)       # (4.1)
        z1 = f * U + (1 - f) * V                # (4.2)
        g = self.σ(z1 @ self.W2 + self.b2)      # (4.3)
        z2 = g * U + (1 - g) * V                # (4.4)
        h = self.σ(z2 @ self.W3 + self.b3)      # (4.5)
        x_new = self.α * h + (1 - self.α) * x   # (4.6)
        return x_new
    
class FinalOutput(nn.Module):
    def __init__(self, neurons, out_dim):
        super().__init__()  # inherit
        self.W = nn.Parameter(torch.randn(neurons, out_dim, requires_grad=True))

        # Xavier Initialization
        init.xavier_uniform_(self.W)

    def forward(self, x): return x @ self.W

# PINN Architecture (α_inity -> u, ϕ)
blocks = [PirateNetBlock(in_dim=1, neurons=config.NEURONS, scale=config.SCALE, α_init=config.α_INIT)]
for block in range(config.BLOCKS - 1): blocks.extend([PirateNetBlock(in_dim=config.NEURONS, neurons=config.NEURONS, scale=config.SCALE, α_init=config.α_INIT)])
blocks.append(FinalOutput(neurons=config.NEURONS, out_dim=2))
PINN = nn.Sequential(*blocks)

@dataclass 
class Trials:
    u_trial: torch.Tensor
    ϕ_trial: torch.Tensor

def generate_trials(params) -> Trials:
    # Trial functions
    u_trial = lambda y: PINN(torch.cat([y], dim=1))[:,0:1] * (1 + y) * (1 - y)  # torch.Size([y, 1])
    ϕ_trial = lambda y: params.ϕ_max * torch.sigmoid(PINN(torch.cat([y], dim=1)))[:,1:2] * (1 + y) * (1 - y)  # torch.Size([y, 1])

    return Trials(
        u_trial=u_trial,
        ϕ_trial=ϕ_trial,
    )
    
