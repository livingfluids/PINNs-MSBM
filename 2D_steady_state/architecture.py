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
    
class MultiScaleFourierFeatures(nn.Module):
    def __init__(self, in_dim, neurons, scales):
        super().__init__()
        self.blocks = nn.ModuleList([
            FourierFeatures(in_dim, neurons, scale) for scale in scales
        ])

    def forward(self, x):
        return torch.cat([block(x) for block in self.blocks], dim=-1)

# Simple PINN Architecture
def buildSimplePINN(neurons):
    return nn.Sequential(

        # 1 Fourier Input Layer
        MultiScaleFourierFeatures(in_dim=2, neurons=neurons, scales=config.SCALES),

        # 3 Hidden Layers
        nn.Linear(neurons * 2 * len(config.SCALES), neurons),
        config.ACTIVATION,

        nn.Linear(neurons, neurons),
        config.ACTIVATION,

        nn.Linear(neurons, neurons),
        config.ACTIVATION,
        
        # 1 Linear Output Layer
        nn.Linear(neurons, 4),  # u, v, p, ϕ
    )

# Trial Function Data Class
@dataclass
class Trials: all_trials: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

"""# Build Trial Functions
def buildTrials(params, device, PINN) -> Trials:
    def all_trials(xy: torch.Tensor):
        # Streamfunction trials require coordinate gradients even in plotting
        # and diagnostics that call this function under torch.no_grad().
        with torch.enable_grad():
            if not xy.requires_grad: xy = xy.detach().requires_grad_(True)
            # - 
            pinn    = PINN(xy)
            ψ       = pinn[:, 0:1]  # unconstrained streamfunction
            p       = pinn[:, 1:2]
            ϕ       = params.ϕ_max * torch.sigmoid(pinn[:, 2:3])
            # - 
            dψ = torch.autograd.grad(ψ.sum(), xy, create_graph=True, retain_graph=True)[0]
            u =  dψ[:, 1:2]   # ∂ψ/∂y
            v = -dψ[:, 0:1]   # -∂ψ/∂x
            # - 
        return u, v, p, ϕ
    return Trials(all_trials=all_trials)"""

# Build Trial Functions
def buildTrials(params, device, PINN) -> Trials:
    def all_trials(xy: torch.Tensor):
        pinn    = PINN(xy)
        # - 
        u       = pinn[:, 0:1]
        v       = pinn[:, 1:2]
        p       = pinn[:, 2:3]
        ϕ       = pinn[:, 3:4]
        # - 
        return u, v, p, ϕ
    return Trials(all_trials=all_trials)
