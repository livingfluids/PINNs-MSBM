import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from pathlib import Path
import pandas as pd
import config as c
import yaml
import os
from dataclasses import dataclass

# To-Do:
# save model code for all cases 
# correct print loss value
# range + 1, but still omit LBFG is epochs are set to 0

# Device 
if c.use_GPU:
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
else: device = torch.device("cpu")
print(f"Using device: {device}")

# initialize
ℒ_J_init, ℒ_Σxy_init, ℒ_Σyy_init, ℒ_mass_init, ℒ_symmetry_init, ℒ_data_init, ℒ_init = None, None, None, None, None, None, None

# Seed for reproducability 
torch.manual_seed(0)

# Paths
ROOT = Path(__file__).parent
data_dir    = ROOT / c.data_path / c.data_file_directory
visuals_dir = ROOT / f"{c.visuals_path}_{c.data_file_directory}"
models_dir  = ROOT / c.models_path
visuals_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# Load the data
df = pd.read_csv(data_dir / 'data.csv')
with open(data_dir / 'parameters.yaml', 'r') as f: config_data = yaml.safe_load(f)

# Physical parameters 
Ux_max = torch.tensor(df['u'].values.max(), device=device, dtype=torch.float32)  # max steady state velocity (m/s)
ϕ_max = torch.tensor(config_data['phi_max'], device=device, dtype=torch.float32)  # max ϕ (dimensionless)
ϕ_bulk = torch.tensor(config_data['phi_bulk'], dtype=torch.float32, device=device)  # bulk ϕ
H = torch.tensor(config_data['H'], device=device, dtype=torch.float32)  # channel height (m) 
ρ = torch.tensor(config_data['rho'], device=device, dtype=torch.float32)  # solvent density (Kg/m³)
η = torch.tensor(config_data['eta'], device=device, dtype=torch.float32)  # dynamic viscosity (Pa·s)
η0 = η / ρ  # kinematic viscosity (m²/s)  
Kn = torch.tensor(config_data['Kn'], device=device, dtype=torch.float32)  # fitting parameter (dimensionless)
λ2 = torch.tensor(config_data['lambda2'], device=device, dtype=torch.float32)  # fitting parameter (dimensionless)
λ3 = torch.tensor(config_data['lambda3'], device=device, dtype=torch.float32)  # fitting parameter (dimensionless)
α = torch.tensor(config_data['alpha'], device=device, dtype=torch.float32)  # fitting parameter α ∈ [2, 5] (dimensionless)
if c.β_learnable: β = nn.Parameter(torch.tensor([1.0], device=device, requires_grad=True) * c.β_initial_guess)  # torch.Size([y, 1])
else: β = torch.tensor(config_data['beta'], device=device, dtype=torch.float32)  # power-law coefficient
a = torch.tensor(config_data['a'], device=device, dtype=torch.float32)  # particle radius (m)
ε = a / ((H / 2)**2)  # non-local shear-rate coefficient (1/m)
frv = torch.tensor(config_data['frv'], device=device, dtype=torch.float32)  # function of the reduced volume
p = torch.tensor(config_data['p'], device=device, dtype=torch.float32)
dpstar_dxstar = nn.Parameter(torch.tensor([1], dtype=torch.float32, device=device))  # normalized x-pressure gradient (dimensionless)
H0 = torch.tensor(config_data['H0'], device=device, dtype=torch.float32)

# Data tensors
y_data = 2.0 * torch.tensor(df['y'].values, dtype=torch.float32, device=device).unsqueeze(1) / H - 1.0
u_data = torch.tensor(df['u'].values, dtype=torch.float32, device=device).unsqueeze(1) / Ux_max
if 'phi' in df.columns: ϕ_data = torch.tensor(df['phi'].values, dtype=torch.float32, device=device).unsqueeze(1)
else: ϕ_data = None

# Collocation points 
if c.wall_skewed_collocation: y = torch.tanh(3 * torch.linspace(-1.0, 1.0, c.collocation_points + 2, device=device).unsqueeze(1).requires_grad_(True)[1:-1])
else: y = torch.linspace(-1.0, 1.0, c.collocation_points + 2, device=device).unsqueeze(1).requires_grad_(True)[1:-1]

# Call these hyperarameters from the config once, so that they aren't called repeatedly during training---helps reduce compute
use_FDM = c.use_FDM
PINN_collocation_points = c.collocation_points
save_images = c.save_images
single_epochs_ADAM = c.single_PINN_epochs_ADAM
single_epochs_LBFGS = c.single_PINN_epochs_LBFGS
u_epochs_ADAM = c.u_PINN_epochs_ADAM
u_epochs_LBFGS = c.u_PINN_epochs_LBFGS
ϕ_epochs_ADAM = c.ϕ_PINN_epochs_ADAM
ϕ_epochs_LBFGS = c.ϕ_PINN_epochs_LBFGS
use_scheduler = c.use_scheduler
visualize_step = c.visualize_step
sequential_training = c.sequential_training
β_learnable = c.β_learnable
sqrt_losses = c.sqrt_losses
use_spatially_adaptive_learnable_parameters = c.use_spatially_adaptive_learnable_parameters
two_PINNs = c.two_PINNs
β_true = torch.tensor(config_data['beta'], device=device, dtype=torch.float32)
normalize_losses = c.normalize_losses
sqrt_data_loss = c.sqrt_data_loss

# Trial functions
if c.two_PINNs:
    u_trial = lambda y: u_PINN(torch.cat([y.to(device)], dim=1))[:,0:1] * (1 + y.to(device)) * (1 - y.to(device)) if c.u_zero_Dirichlet_at_walls else u_PINN(torch.cat([y.to(device)], dim=1))[:,0:1]  # torch.Size([y, 1]), normalized 
    ϕ_trial = lambda y: ϕ_max * torch.sigmoid(ϕ_PINN(y.to(device))[:,0:1]) * (1 + y.to(device)) * (1 - y.to(device)) if c.ϕ_zero_Dirichlet_at_walls else ϕ_max * torch.sigmoid(ϕ_PINN(y.to(device))[:,0:1])  # torch.Size([y, 1]), sigmoid keeps bounded between 0 and ϕ_max
else: 
    u_trial = lambda y: PINN(torch.cat([y.to(device)], dim=1))[:,0:1] * (1 + y.to(device)) * (1 - y.to(device)) if c.u_zero_Dirichlet_at_walls else PINN(torch.cat([y.to(device)], dim=1))[:,0:1]  # torch.Size([y, 1]), normalized 
    ϕ_trial = lambda y: ϕ_max * torch.sigmoid(PINN(y.to(device))[:,1:2]) * (1 + y.to(device)) * (1 - y.to(device)) if c.ϕ_zero_Dirichlet_at_walls else ϕ_max * torch.sigmoid(PINN(y.to(device))[:,1:2])  # torch.Size([y, 1]), sigmoid keeps bounded between 0 and ϕ_max

# Loss history for plotting
class LossHistory:
    def __init__(self):
        self.total = []
        self.individuals = []
        self.β = []

    def append(self, ℒ, ℒ_individuals, β):
        self.total.append(ℒ.item())
        self.individuals.append([ℒ_individual.item() for ℒ_individual in ℒ_individuals])
        self.β.append(β.item())

# PINN ------------------------------------------------------------------------ 
# Modified Gaussian Expansion 
class ModifiedGaussianExpansion(nn.Module):
    def __init__(self, neurons, scale):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, neurons) * scale)
        self.beta = nn.Parameter(torch.randn(1, neurons) * scale)
        self.gamma = nn.Parameter(torch.randn(1, neurons) * scale)
        self.kappa = nn.Parameter(torch.randn(1, neurons))

    def forward(self, x):
        gauss_function = torch.abs(self.alpha) * torch.exp(-torch.abs(self.beta) * torch.abs((torch.abs(self.kappa) + 0.1) * x)**(torch.abs(self.gamma) + 1))
        return gauss_function

# Fourier features from Tancik et al. 
class FourierFeatures(nn.Module):
    def __init__(self, neurons, scale):
        super().__init__()
        self.B = nn.Parameter(torch.randn(1, neurons) * scale)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# Both Fourier features and Modified Gaussian Expansion
class Both(nn.Module):
    def __init__(self, neurons, gauss_scale, fourier_scale):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, neurons) * gauss_scale)
        self.beta = nn.Parameter(torch.randn(1, neurons) * gauss_scale)
        self.gamma = nn.Parameter(torch.randn(1, neurons) * gauss_scale)
        self.kappa = nn.Parameter(torch.randn(1, neurons))

        self.B = nn.Parameter(torch.randn(1, neurons) * fourier_scale)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        gauss_function = torch.abs(self.alpha) * torch.exp(-torch.abs(self.beta) * torch.abs((torch.abs(self.kappa) + 0.1) * x)**(torch.abs(self.gamma) + 1))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj), gauss_function], dim=-1)

# Compile the PINN or PINNs
if c.two_PINNs:
    # Compile Ux_PINN layers
    if c.u_PINN_Fourier_scale != 0: u_layers = [FourierFeatures(neurons=c.u_PINN_neurons, scale=c.u_PINN_Fourier_scale), nn.Linear(2 * c.u_PINN_neurons, c.u_PINN_neurons), c.u_PINN_activation_function]
    else: u_layers = [nn.Linear(1, c.u_PINN_neurons), c.u_PINN_activation_function]
    for layer in range(c.u_PINN_layers): u_layers.extend([nn.Linear(c.u_PINN_neurons, c.u_PINN_neurons), c.u_PINN_activation_function])
    u_layers.append(nn.Linear(c.u_PINN_neurons, 1))

    # Create the Ux_PINN
    u_PINN = nn.Sequential(*u_layers).to(device)

    # Compile ϕ_PINN layers 
    if c.ϕ_PINN_Fourier_scale != 0 and c.ϕ_PINN_Gauss_scale == 0: ϕ_layers = [FourierFeatures(neurons=c.ϕ_PINN_neurons, scale=c.ϕ_PINN_Fourier_scale), nn.Linear(2 * c.ϕ_PINN_neurons, c.ϕ_PINN_neurons), c.ϕ_PINN_activation_function]
    elif c.ϕ_PINN_Fourier_scale == 0 and c.ϕ_PINN_Gauss_scale != 0: ϕ_layers = [ModifiedGaussianExpansion(neurons=c.ϕ_PINN_neurons, scale=c.ϕ_PINN_Gauss_scale), nn.Linear(c.ϕ_PINN_neurons, c.ϕ_PINN_neurons), c.ϕ_PINN_activation_function]
    elif c.ϕ_PINN_Fourier_scale != 0 and c.ϕ_PINN_Gauss_scale != 0: ϕ_layers = [Both(neurons=c.ϕ_PINN_neurons, fourier_scale=c.ϕ_PINN_Fourier_scale, gauss_scale=c.ϕ_PINN_Gauss_scale), nn.Linear(3 * c.ϕ_PINN_neurons, c.ϕ_PINN_neurons), c.ϕ_PINN_activation_function]
    else: ϕ_layers = [nn.Linear(1, c.ϕ_PINN_neurons), c.ϕ_PINN_activation_function]
    for layer in range(c.ϕ_PINN_layers): ϕ_layers.extend([nn.Linear(c.ϕ_PINN_neurons, c.ϕ_PINN_neurons), c.ϕ_PINN_activation_function])
    ϕ_layers.append(nn.Linear(c.ϕ_PINN_neurons, 1))

    # Create the ϕ_PINN
    ϕ_PINN = nn.Sequential(*ϕ_layers).to(device)
else:
    # Compile PINN layers 
    if c.single_PINN_Fourier_scale != 0 and c.single_PINN_Gauss_scale == 0: layers = [FourierFeatures(neurons=c.single_PINN_neurons, scale=c.single_PINN_Fourier_scale), nn.Linear(2 * c.single_PINN_neurons, c.single_PINN_neurons), c.single_PINN_activation_function]
    elif c.single_PINN_Fourier_scale == 0 and c.single_PINN_Gauss_scale != 0: layers = [ModifiedGaussianExpansion(neurons=c.single_PINN_neurons, scale=c.single_PINN_Gauss_scale), nn.Linear(c.single_PINN_neurons, c.single_PINN_neurons), c.single_PINN_activation_function]
    elif c.single_PINN_Fourier_scale != 0 and c.single_PINN_Gauss_scale != 0: layers = [Both(neurons=c.single_PINN_neurons, fourier_scale=c.single_PINN_Fourier_scale, gauss_scale=c.single_PINN_Gauss_scale), nn.Linear(3 * c.single_PINN_neurons, c.single_PINN_neurons), c.single_PINN_activation_function]
    else: layers = [nn.Linear(1, c.single_PINN_neurons), c.single_PINN_activation_function]
    for layer in range(c.single_PINN_layers): layers.extend([nn.Linear(c.single_PINN_neurons, c.single_PINN_neurons), c.single_PINN_activation_function])
    layers.append(nn.Linear(c.single_PINN_neurons, 2))

    # Create the PINN
    PINN = nn.Sequential(*layers).to(device)

# Self-Adaptive Weights -------------------------------------------------------
# Self-Adaptive Weights from McClenny & Braga-Neto 
mask = c.λ_mask

λ_J = nn.Parameter(torch.ones([c.collocation_points, 1], device=device, requires_grad=True) * c.initial_λ_J) if use_spatially_adaptive_learnable_parameters else 1  # torch.Size([y, 1])
λ_Σxy = nn.Parameter(torch.ones([c.collocation_points, 1], device=device, requires_grad=True) * c.initial_λ_Σxy) if use_spatially_adaptive_learnable_parameters else 1  # torch.Size([y, 1])
λ_Σyy = nn.Parameter(torch.ones([c.collocation_points, 1], device=device, requires_grad=True) * c.initial_λ_Σyy) if use_spatially_adaptive_learnable_parameters else 1  # torch.Size([y, 1])
λ_mass = nn.Parameter(torch.ones([1], device=device, requires_grad=True) * c.initial_λ_mass) if use_spatially_adaptive_learnable_parameters else 1  # torch.Size([1])
λ_symmetry = nn.Parameter(torch.ones([1], device=device, requires_grad=True) * c.initial_λ_symmetry) if use_spatially_adaptive_learnable_parameters else 1  # torch.Size([1])
λ_data = nn.Parameter(torch.ones([y_data.size()[0], 1], device=device, requires_grad=True) * c.initial_λ_data) if use_spatially_adaptive_learnable_parameters else 1  # torch.Size([y_data, 1])

# Helper Functions ------------------------------------------------------------
def gradient(y, f):
    y = y if y.ndim == 2 else y.unsqueeze(1)

    if use_FDM:
        Δy = 2 / PINN_collocation_points
        dfdystar_center = (f[2:] - f[:-2]) / (2 * Δy)
        dfdystar_left = (f[1] - f[0]) / Δy 
        dfdystar_right = (f[-1] - f[-2]) / Δy
        dfdystar = torch.cat([dfdystar_left.unsqueeze(0), dfdystar_center, dfdystar_right.unsqueeze(0)], dim=0)  # torch.Size([y, 1])   
    else: dfdystar = torch.autograd.grad(f, y, torch.ones_like(f), create_graph=True)[0]  # torch.Size([y, 1])

    return dfdystar

def current_lr(opt: torch.optim.Optimizer) -> float: return opt.param_groups[0]["lr"]

@dataclass
class OptimizerScheduler:
    optimizer_adam: torch.optim.Optimizer
    optimizer_lbfgs: torch.optim.Optimizer
    
    scheduler_adam: torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_lbfgs: torch.optim.lr_scheduler.ReduceLROnPlateau

def make_optimizer_scheduler(parameters, learning_rate) -> OptimizerScheduler:
    optimizer_adam = torch.optim.Adam(parameters, lr=learning_rate)
    optimizer_lbfgs = torch.optim.LBFGS(parameters, lr=learning_rate)
    
    scheduler_adam = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, factor=c.scheduler_factor, patience=c.scheduler_patience, min_lr=c.scheduler_min_learning_rate)
    scheduler_lbfgs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lbfgs, factor=c.scheduler_factor, patience=c.scheduler_patience, min_lr=c.scheduler_min_learning_rate)
    
    return OptimizerScheduler(optimizer_adam, optimizer_lbfgs, scheduler_adam, scheduler_lbfgs)

# Loss Functions --------------------------------------------------------------
def u_data_loss():
    u_pred = u_trial(y_data)

    u_data_term = u_pred - u_data
    
    return u_data_term

def physics_loss():  # physics ensures ∇⋅J = ∇⋅Σ = 0
    ystar = y  # already normalized 
    if use_FDM: Δystar = 2 / PINN_collocation_points
    ustar = u_trial(ystar)  # already normalized 
    ϕ = ϕ_trial(ystar)
    A = 2 * a / H
    pstar = p * H / (2 * η0 * Ux_max)
    zero = torch.zeros_like(ystar, device=device)  # torch.Size([y, 1])

    # Normal stress viscosity (ηₙ(ϕ))
    def ηN(ϕ):
        return Kn * (ϕ/ϕ_max)**2 * (1 - ϕ/ϕ_max)**(-2)  # torch.Size([y, 1]), a scalar for each y

    # Shear viscosity of the particle phase (ηₚ(ϕ))
    def ηp(ϕ):
        ηs = (1 - ϕ/ϕ_max)**(-2)
        return ηs - 1  # torch.Size([y, 1]), a scalar for each y

    # Sedimentation hinderence function for mobility of particle phase (f(ϕ))
    def f(ϕ):
        return (1 - ϕ/ϕ_max) * (1 - ϕ)**(α - 1)  # torch.Size([y, 1]), a scalar for each y

    # Gradient of the velocity field (∇U)
    dUxstar_dystar = gradient(ystar, ustar)  # torch.Size([y, 1])
    Ustar_gradient = torch.stack([
        torch.cat([zero, dUxstar_dystar, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Strain rate tensor (E)
    Estar = 0.5 * (Ustar_gradient + Ustar_gradient.transpose(1, 2))  # torch.Size([y, 3, 3]), a matrix for each y

    # Shear rate tensor (γ̇)
    γ̇star = torch.sqrt(2 * torch.sum(Estar * Estar, dim=(1, 2))).unsqueeze(1)  # torch.Size([y, 1])

    # Lift force (L)
    γ̇ = γ̇star * 2 * Ux_max / H  # dimensionalize for calculating it
    left_wall = 3 * η0 * γ̇ / (4 * torch.pi * ((H/2)*(ystar + 1) + H0)**β) * frv
    right_wall = 3 * η0 * γ̇ / (4 * torch.pi * ((H/2)*(1 - ystar) + H0)**β) * frv
    scale_L = (H ** 2) / (2 * η0 * Ux_max)  # nondimensionalize after calculating it 
    L = torch.stack([
        torch.cat([zero], dim=1),
        torch.cat([scale_L * (left_wall - right_wall)], dim=1),
        torch.cat([zero], dim=1)
    ], dim=1)
    lift_force_visualize = torch.cat([(left_wall - right_wall)], dim=1).detach().cpu().numpy()[:, 0]
    
    # Diagonal tensor of the SBM (Q)
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, λ2, 0.0], [0.0, 0.0, λ3]], device=device).repeat(ystar.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Non-local shear rate tensor
    γ̇NLstar = ε * H / 2

    # Particle normal stress diagonal tensor (Σₙₙᵖ)
    Σpnnstar = ηN(ϕ).view(-1, 1, 1) * (γ̇star.unsqueeze(1) + γ̇NLstar) * Q  # torch.Size([y, 3, 3]), a matrix for each y

    # Oriented particle stress tensor (Σᵖ)
    Σpstar = -Σpnnstar + (2 * ηp(ϕ).view(-1, 1, 1) * Estar)  # torch.Size([y, 3, 3]), a matrix for each y

    # Divergence of oriented particle stress tensor (∇⋅Σᵖ)
    dΣpxystar_dystar = gradient(ystar, Σpstar[:, 0, 1])  # torch.Size([y, 1])
    dΣpyystar_dystar = gradient(ystar, Σpstar[:, 1, 1])  # torch.Size([y, 1])
    Σpstar_divergence = torch.stack([
        torch.cat([zero + dΣpxystar_dystar + zero], dim=1),
        torch.cat([zero + dΣpyystar_dystar + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y

    # Migration flux (J)
    Jstar = - (2 * A**2 / 9) * f(ϕ).unsqueeze(1) * (Σpstar_divergence + ϕ.view(-1, 1, 1) * L)  # torch.Size([y, 3, 1])

    # Divergence of migration flux (∇⋅J)
    dJxstar_dxstar = dJzstar_dzstar = zero
    dJystar_dystar = gradient(ystar, Jstar[:, 1, 0])  # torch.Size([y, 1])
    Jstar_divergence = dJxstar_dxstar + dJystar_dystar + dJzstar_dzstar  # torch.Size([y, 1])

    # Identity matrix (I)
    I = torch.eye(3, device=device).repeat(ystar.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid phase stress (Σᶠ)
    Σfstar = - pstar * I + 2 * Estar

    # Total stress (Σ)
    Σstar = Σpstar + Σfstar

    # Suspension momentum balance (∇⋅Σ)
    dΣxystar_dystar = gradient(ystar, Σstar[:, 0, 1])  # torch.Size([y, 1])
    dΣyystar_dystar = gradient(ystar, Σstar[:, 1, 1])  # torch.Size([y, 1])

    return Jstar_divergence, dΣxystar_dystar, dΣyystar_dystar, lift_force_visualize

def ϕ_bulk_loss():  # IC ensures mean(ϕ) never changes
    ystar = y   # already normalized
    ustar = u_trial(ystar)  # already normalized 
    ϕ = ϕ_trial(ystar)

    # Mass conservation error calculation
    ϕ_bulk_term = torch.sum(ϕ * ustar) / torch.sum(ustar) - ϕ_bulk 

    return ϕ_bulk_term

def ϕ_symmetry_loss():  # ensures ϕ is symmetric along centerflow axis
    ystar = y   # already normalized

    # Symmetry error calculation 
    ϕ_symmetry_term = ϕ_trial(ystar) - ϕ_trial(-ystar)

    return ϕ_symmetry_term

def total_loss(J_global, Σxy_global, Σyy_global, mass_global, symmetry_global, data_global):  # combining losses
    global ℒ_J_init, ℒ_Σxy_init, ℒ_Σyy_init, ℒ_mass_init, ℒ_symmetry_init, ℒ_data_init, ℒ_init
    
    if (J_global + Σxy_global + Σyy_global != 0) or (ℒ_J_init is None and ℒ_Σxy_init is None and ℒ_Σyy_init is None): use_physics = True
    else: use_physics = False
    
    if use_physics: Jstar_divergence, dΣxystar_dystar, dΣyystar_dystar, lift_force_visualize = physics_loss()  # torch.Size([y, 1]) for each
    else: Jstar_divergence, dΣxystar_dystar, dΣyystar_dystar, lift_force_visualize = 0, 0, 0, None
    ϕ_bulk_term = ϕ_bulk_loss()  # torch.Size([1])
    ϕ_symmetry_term = ϕ_symmetry_loss()  # torch.Size([1])
    u_data_term = u_data_loss()  # torch.Size([y_data, 1])

    # Indivisual losses, in the style of McClenny & Braga-Neto, all become scalars
    ℒ_J = torch.mean(mask(λ_J) * Jstar_divergence**2) if use_physics else ℒ_J_init
    ℒ_Σxy = torch.mean(mask(λ_Σxy) * dΣxystar_dystar**2) if use_physics else ℒ_Σxy_init
    ℒ_Σyy = torch.mean(mask(λ_Σyy) * dΣyystar_dystar**2) if use_physics else ℒ_Σyy_init
    ℒ_mass = torch.mean(mask(λ_mass) * ϕ_bulk_term**2)
    ℒ_symmetry = torch.mean(mask(λ_symmetry) * ϕ_symmetry_term**2)
    ℒ_data = torch.mean(mask(λ_data) * u_data_term**2)
    # NOTE ℒ_mass does not need mean(...), as it is already scalars, but for the the sake of code consistiency, they are

    # normalize losses
    if normalize_losses:
        if ℒ_J_init is None: ℒ_J_init = ℒ_J.detach()
        if ℒ_Σxy_init is None: ℒ_Σxy_init = ℒ_Σxy.detach()
        if ℒ_Σyy_init is None: ℒ_Σyy_init = ℒ_Σyy.detach() 
        if ℒ_mass_init is None: ℒ_mass_init = ℒ_mass.detach()
        if ℒ_symmetry_init is None: ℒ_symmetry_init = ℒ_symmetry.detach()
        if ℒ_data_init is None: ℒ_data_init = ℒ_data.detach()
        if ℒ_init is None: ℒ_init = ℒ_J_init + ℒ_Σxy_init + ℒ_Σyy_init + ℒ_mass_init + ℒ_symmetry_init + ℒ_data_init

        ℒ_J = ℒ_J / ℒ_init
        ℒ_Σxy = ℒ_Σxy / ℒ_init
        ℒ_Σyy = ℒ_Σyy / ℒ_init
        ℒ_mass = ℒ_mass / ℒ_init
        ℒ_symmetry = ℒ_symmetry / ℒ_init
        ℒ_data = ℒ_data / ℒ_init

    # loss function, in the style of McClenny & Braga-Neto
    if sqrt_losses: ℒ = J_global * torch.sqrt(ℒ_J) + Σxy_global * torch.sqrt(ℒ_Σxy) + Σyy_global * torch.sqrt(ℒ_Σyy) + mass_global * torch.sqrt(ℒ_mass) + symmetry_global * torch.sqrt(ℒ_symmetry) + data_global * torch.sqrt(ℒ_data)
    elif sqrt_data_loss: ℒ = J_global * ℒ_J + Σxy_global * ℒ_Σxy + Σyy_global * ℒ_Σyy + mass_global * ℒ_mass + symmetry_global * ℒ_symmetry + data_global * torch.sqrt(ℒ_data)
    else: ℒ = J_global * ℒ_J + Σxy_global * ℒ_Σxy + Σyy_global * ℒ_Σyy + mass_global * ℒ_mass + symmetry_global * ℒ_symmetry + data_global * ℒ_data
 
    # Individual losses for tracking and eventual visualization for debugging
    ℒ_individuals = [ℒ_J, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_symmetry, ℒ_data]

    # Visuals | sqrt is NOT applied to them, since it is an optimization trick
    ℒ_visualize = ℒ_J + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_symmetry + ℒ_data
    ℒ_individuals_visualize = ℒ_J, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_symmetry, ℒ_data
 
    return ℒ, ℒ_visualize, ℒ_individuals, ℒ_individuals_visualize, lift_force_visualize

# Visualize -------------------------------------------------------------------
def visualize(epoch, lift_force_visualize):
    with torch.no_grad():
        y_plot = torch.linspace(-1.0, 1.0, PINN_collocation_points, device=device).unsqueeze(1)
        y_plot_dim = ((y_plot + 1.0) / 2.0 * H).cpu().numpy()
        Ux_pinn = (u_trial(y_plot) * Ux_max).cpu().numpy()
        phi_pinn = ϕ_trial(y_plot).cpu().numpy()

        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        ax_ux = axs[0][0]
        ax_phi = axs[0][1]
        ax_total_loss = axs[1][1]
        ax_indiv_loss = axs[1][0]
        ax_L = axs[0][2]
        ax_β = axs[1][2]

        # Ux
        ax_ux.plot(((y_data + 1)/2 * H).cpu(), (u_data * Ux_max).cpu(), 'ko', markersize=3, label='Data')
        ax_ux.plot(y_plot_dim, Ux_pinn, 'b-', label='PINN')
        ax_ux.set_xlabel('y [m]')
        ax_ux.set_ylabel('Ux [m/s]')
        ax_ux.legend()
        ax_ux.grid(True)

        # Phi
        if ϕ_data is not None:
            ax_phi.plot(((y_data + 1)/2 * H).cpu(), ϕ_data.cpu(), 'ko', markersize=3, label='Data')
        ax_phi.plot(y_plot_dim, phi_pinn, 'r-', label='PINN')
        ax_phi.set_xlabel('y [m]')
        ax_phi.set_ylabel('phi')
        ax_phi.legend()
        ax_phi.grid(True)

        # Total Loss
        if loss_history.total:
            ax_total_loss.semilogy(loss_history.total, 'g-', label='Total Loss')
            ax_total_loss.set_xlabel('Epoch')
            ax_total_loss.set_ylabel('Weighted Loss')
            ax_total_loss.legend()
            ax_total_loss.grid(True)

        # Individual Losses plot
        if loss_history.individuals: # ∇⋅J = ∇⋅Σ
            component_names = ["∇⋅J", "∇⋅Σ (xy)", "∇⋅Σ (yy)", "IC", "sym", "data"]
            for i, indiv_loss in enumerate(zip(*loss_history.individuals)):
                ax_indiv_loss.semilogy(indiv_loss, label=f'Loss {component_names[i]}')
            ax_indiv_loss.set_xlabel('Epoch')
            ax_indiv_loss.set_ylabel('Weighted Losses')
            ax_indiv_loss.legend()
            ax_indiv_loss.grid(True)

        # β plot
        if loss_history.β:
            ax_β.semilogy(loss_history.β, 'g-', label='β (learned)')
            if β_true is not None: ax_β.axhline(β_true.item(), color='r', linestyle='--', label=f'True β = {β_true.item():.4f}')
            ax_β.set_xlabel('Epoch')
            ax_β.set_ylabel('β')
            ax_β.legend()
            ax_β.grid(True)

        # Lift force plot
        if lift_force_visualize is not None:
            ax_L.plot(y_plot_dim, lift_force_visualize, 'y-', label='Lift Force')
            ax_L.set_xlabel('y [m]')
            ax_L.set_ylabel('Lift Force')
            ax_L.legend()
            ax_L.grid(True)

        # For animation
        visualization_history.append({
            'epoch': epoch,
            'y_plot_dim': y_plot_dim,
            'phi_pinn': phi_pinn
        })

        plt.tight_layout()
        plt.show(block=False)
        if save_images: plt.savefig(visuals_dir / f"plot_epoch_{epoch}.png")
        plt.pause(0.1)
        plt.close(fig)

def animate_mp4():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel('y [m]')
    ax.set_ylabel('phi')
    ax.set_xlim(0, H.cpu().numpy())
    ax.grid(True)
    line_phi_data, = ax.plot(((y_data + 1)/2 * H).cpu(), ϕ_data.cpu(), 'ko', markersize=3, label='Data')
    line_phi_pinn, = ax.plot([], [], 'r-', label='PINN')
    ax.legend()
    title = ax.set_title('')

    def init():
        line_phi_pinn.set_data([], [])
        title.set_text('')
        return [line_phi_pinn, title]
    
    def update(frame):
        data = visualization_history[frame]
        line_phi_pinn.set_data(data['y_plot_dim'], data['phi_pinn'])
        title.set_text(f'Epoch {data["epoch"]}')

        return [line_phi_pinn, title]
    
    anim = animation.FuncAnimation(fig, update, frames=len(visualization_history), init_func=init, blit=True, interval=100)
    anim.save(visuals_dir / c.gif_name, writer='ffmpeg', fps=10)
    plt.close(fig)

# Training Loop -------------------------------------------------

# ---------------------------------------------------------------
# Two PINNs
# ---------------------------------------------------------------
if c.two_PINNs:
    # Define parameters
    u_PINN_parameters = list(u_PINN.parameters())
    ϕ_PINN_parameters = list(ϕ_PINN.parameters()) + [dpstar_dxstar]
    if use_spatially_adaptive_learnable_parameters:
        u_λ_parameters = [λ_data]
        ϕ_λ_parameters = [λ_J, λ_Σxy, λ_Σyy, λ_mass, λ_symmetry]
    if c.β_learnable: β_parameter = [β]

    # Define optimizers & schedulers 
    u_PINN_optimizer_scheduler = make_optimizer_scheduler(u_PINN_parameters, c.u_PINN_learning_rate)
    ϕ_PINN_optimizer_scheduler = make_optimizer_scheduler(ϕ_PINN_parameters, c.ϕ_PINN_learning_rate)
    if use_spatially_adaptive_learnable_parameters:
        u_λ_optimizer_scheduler = make_optimizer_scheduler(u_λ_parameters, c.u_PINN_λ_learning_rate)
        ϕ_λ_optimizer_scheduler= make_optimizer_scheduler(ϕ_λ_parameters, c.ϕ_PINN_λ_learning_rate)
    if c.β_learnable: β_optimizer_scheduler = make_optimizer_scheduler(β_parameter, c.β_learning_rate)

    # Histories
    loss_history = LossHistory()
    visualization_history = []

    # ---------------------------------------------------------------
    # Sequential Training
    # ---------------------------------------------------------------
    if sequential_training:
        if c.use_saved_u_PINN_model:
            u_saved_path = models_dir / f"u_model_{c.data_file_directory}"
            u_PINN.load_state_dict(torch.load(f=u_saved_path))
            u_PINN.eval()

        # ---------------------------------------------------------------
        # Train u
        # ---------------------------------------------------------------
        else:
            # u ADAM training loop
            if u_epochs_ADAM != 0:
                for epoch in range(u_epochs_ADAM + 1):
                    u_PINN_optimizer_scheduler.optimizer_adam.zero_grad()
                    if use_spatially_adaptive_learnable_parameters: u_λ_optimizer_scheduler.optimizer_adam.zero_grad()

                    # Forward & Backward passes
                    ℒ, ℒ_visualize, ℒ_individuals, ℒ_individuals_visualize, lift_force_visualize = total_loss(J_global=0, Σxy_global=0, Σyy_global=0, mass_global=0, symmetry_global=0, data_global=1)
                    ℒ.backward()  # backward pass to compute gradients

                    # Gradient descent & scheduler update
                    u_PINN_optimizer_scheduler.optimizer_adam.step()  # gradient descent updating PINN parameters
                    if use_scheduler: u_PINN_optimizer_scheduler.scheduler_adam.step(ℒ.item())

                    # Gradient ascent & scheduler update
                    if use_spatially_adaptive_learnable_parameters: 
                        for λ in u_λ_parameters:
                            if λ.grad is not None:
                                λ.grad = -λ.grad
                        u_λ_optimizer_scheduler.optimizer_adam.step()  # gradient ascent updating self-adaptive weights
                        if use_scheduler: u_λ_optimizer_scheduler.scheduler_adam.step(ℒ.item())

                    # Update loss history for plotting
                    loss_history.append(ℒ_visualize, ℒ_individuals_visualize, β)

                    # Visualize
                    extra = f" | β {β.item():.3f}" if c.β_learnable else ""
                    print(f"u-ADAM ep {epoch} | ℒ {ℒ_visualize.item():.6g} | u_lr {current_lr(u_PINN_optimizer_scheduler.optimizer_adam):.2e}{extra}")
                    print("dp*: ", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
                    if epoch % visualize_step == 0: visualize(epoch, lift_force_visualize)
            
            # u LBFGS training loop
            if u_epochs_LBFGS != 0:
                def closure_u():
                    u_PINN_optimizer_scheduler.optimizer_lbfgs.zero_grad()
                    L, _, _, _, _ = total_loss(J_global=0, Σxy_global=0, Σyy_global=0, mass_global=0, symmetry_global=0, data_global=1)
                    L.backward()
                    return L

                for epoch in range(u_epochs_LBFGS + 1):
                    u_PINN_optimizer_scheduler.optimizer_lbfgs.step(closure_u)
                    ℒ, ℒ_visualize, ℒ_individuals, ℒ_individuals_visualize, lift_force_visualize = total_loss(J_global=0, Σxy_global=0, Σyy_global=0, mass_global=0, symmetry_global=0, data_global=1)
                    if use_scheduler: u_PINN_optimizer_scheduler.scheduler_lbfgs.step(L.item())
                    
                    # Update loss history for plotting
                    loss_history.append(ℒ_visualize, ℒ_individuals_visualize, β)

                    # Visualize
                    extra = f" | β {β.item():.3f}" if c.β_learnable else ""
                    print(f"u-LBFGS ep {epoch} | ℒ {ℒ_visualize.item():.6g} | u_lbfgs_lr {current_lr(u_PINN_optimizer_scheduler.optimizer_lbfgs):.2e}{extra}")
                    print("dp*: ", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
                    if epoch % visualize_step == 0: visualize(epoch, lift_force_visualize)

            # save u_PINN model
            if c.save_u_PINN_model:
                u_saved_path = models_dir / f"u_model_{c.data_file_directory}"
                torch.save(u_PINN.state_dict(), u_saved_path)

        # ---------------------------------------------------------------
        # Train ϕ 
        # ---------------------------------------------------------------
        if not c.train_u_PINN_only:
            # ϕ ADAM training loop
            if ϕ_epochs_ADAM != 0: 
                for epoch in range(ϕ_epochs_ADAM + 1):
                    ϕ_PINN_optimizer_scheduler.optimizer_adam.zero_grad()
                    if use_spatially_adaptive_learnable_parameters: ϕ_λ_optimizer_scheduler.optimizer_adam.zero_grad()
                    if β_learnable: β_optimizer_scheduler.optimizer_adam.zero_grad()

                    # Forward & Backward passes
                    ℒ, ℒ_visualize, ℒ_individuals, ℒ_individuals_visualize, lift_force_visualize = total_loss(J_global=1, Σxy_global=1, Σyy_global=1, mass_global=1, symmetry_global=1, data_global=0)
                    ℒ.backward()  # backward pass to compute gradients

                    # Gradient descent & scheduler update
                    ϕ_PINN_optimizer_scheduler.optimizer_adam.step()  # gradient descent updating PINN parameters
                    if use_scheduler: ϕ_PINN_optimizer_scheduler.scheduler_adam.step(ℒ.item())
                    if β_learnable: 
                        β_optimizer_scheduler.optimizer_adam.step()
                        if use_scheduler: β_optimizer_scheduler.scheduler_adam.step(ℒ.item())

                    # Gradient ascent & scheduler update
                    if use_spatially_adaptive_learnable_parameters: 
                        for λ in ϕ_λ_parameters:
                            if λ.grad is not None:
                                λ.grad = -λ.grad
                        ϕ_λ_optimizer_scheduler.optimizer_adam.step()  # gradient ascent updating self-adaptive weights
                        if use_scheduler: ϕ_λ_optimizer_scheduler.scheduler_adam.step(ℒ.item())

                    # Update loss history for plotting
                    loss_history.append(ℒ_visualize, ℒ_individuals_visualize, β)

                    # Visualize
                    extra = f" | β {β.item():.3f}" if c.β_learnable else ""
                    print(f"ϕ-ADAM ep {epoch} | ℒ {ℒ_visualize.item():.6g} | ϕ_lr {current_lr(ϕ_PINN_optimizer_scheduler.optimizer_adam):.2e}{extra}")
                    print("dp*: ", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
                    if epoch % visualize_step == 0: visualize(epoch, lift_force_visualize)
            
            # ϕ LBFGS training loop
            if ϕ_epochs_LBFGS != 0:
                def closure_u():
                    ϕ_PINN_optimizer_scheduler.optimizer_lbfgs.zero_grad()
                    L, _, _, _, _ = total_loss(J_global=1, Σxy_global=1, Σyy_global=1, mass_global=1, symmetry_global=1, data_global=0)
                    L.backward()
                    return L

                for epoch in range(ϕ_epochs_LBFGS + 1):
                    ϕ_PINN_optimizer_scheduler.optimizer_lbfgs.step(closure_u)
                    ℒ, ℒ_visualize, ℒ_individuals, ℒ_individuals_visualize, lift_force_visualize = total_loss(J_global=1, Σxy_global=1, Σyy_global=1, mass_global=1, symmetry_global=1, data_global=0)
                    if use_scheduler: ϕ_PINN_optimizer_scheduler.scheduler_lbfgs.step(L.item())

                    # Update loss history for plotting
                    loss_history.append(ℒ_visualize, ℒ_individuals_visualize, β)

                    # Visualize
                    extra = f" | β {β.item():.3f}" if c.β_learnable else ""
                    print(f"u-LBFGS ep {epoch} | ℒ {ℒ_visualize.item():.6g} | u_lbfgs_lr {current_lr(u_PINN_optimizer_scheduler.optimizer_lbfgs):.2e}{extra}")
                    print("dp*: ", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
                    if epoch % visualize_step == 0: visualize(epoch, lift_force_visualize)

    # ---------------------------------------------------------------
    # Simultaneous Training
    # ---------------------------------------------------------------
    else:
        # Joint ADAM training loop
        if (u_epochs_ADAM or ϕ_epochs_ADAM) != 0:
            most_epochs_ADAM = max(u_epochs_ADAM, ϕ_epochs_ADAM) + 1
            for epoch in range(most_epochs_ADAM):
                if epoch < u_epochs_ADAM: u_PINN_optimizer_scheduler.optimizer_adam.zero_grad()
                if epoch < ϕ_epochs_ADAM: ϕ_PINN_optimizer_scheduler.optimizer_adam.zero_grad()
                if use_spatially_adaptive_learnable_parameters:
                    if epoch < u_epochs_ADAM: u_λ_optimizer_scheduler.optimizer_adam.zero_grad()
                    if epoch < ϕ_epochs_ADAM: ϕ_λ_optimizer_scheduler.optimizer_adam.zero_grad()
                if β_learnable and epoch < ϕ_epochs_ADAM: β_optimizer_scheduler.optimizer_adam.zero_grad()

                # Forward & Backward passes
                ℒ, ℒ_visualize, ℒ_individuals, ℒ_individuals_visualize, lift_force_visualize = total_loss(J_global=1, Σxy_global=1, Σyy_global=1, mass_global=1, symmetry_global=1, data_global=1)
                ℒ.backward()

                # Gradient descent & scheduler update
                if epoch <= u_epochs_ADAM: u_PINN_optimizer_scheduler.optimizer_adam.step() 
                if epoch <= ϕ_epochs_ADAM: ϕ_PINN_optimizer_scheduler.optimizer_adam.step()
                if use_scheduler:
                    if epoch <= u_epochs_ADAM: u_PINN_optimizer_scheduler.scheduler_adam.step(ℒ.item())
                    if epoch <= ϕ_epochs_ADAM: ϕ_PINN_optimizer_scheduler.scheduler_adam.step(ℒ.item())
                if β_learnable and epoch <= ϕ_epochs_ADAM:
                    β_optimizer_scheduler.optimizer_adam.step()
                    if use_scheduler: β_optimizer_scheduler.scheduler_adam.step(ℒ.item())

                # Gradient ascent & scheduler update
                if use_spatially_adaptive_learnable_parameters: 
                    for λ in (u_λ_parameters + ϕ_λ_parameters):
                        if λ.grad is not None:
                            λ.grad = -λ.grad
                    u_λ_optimizer_scheduler.optimizer_adam.step() 
                    ϕ_λ_optimizer_scheduler.optimizer_adam.step()
                    if use_scheduler: 
                        u_λ_optimizer_scheduler.scheduler_adam.step(ℒ.item())
                        ϕ_λ_optimizer_scheduler.scheduler_adam.step(ℒ.item())

                # Update loss history for plotting
                loss_history.append(ℒ_visualize, ℒ_individuals_visualize, β)
                
                # Visualize
                extra = f" | β {β.item():.3f}" if c.β_learnable else ""
                print(f"joint-ADAM ep {epoch} | ℒ {ℒ_visualize.item():.6g} | u_adam_lr {current_lr(u_PINN_optimizer_scheduler.optimizer_adam):.2e} | ϕ_adam_lr {current_lr(ϕ_PINN_optimizer_scheduler.optimizer_adam):.2e}{extra}")
                print("dp*:", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
                if epoch % visualize_step == 0: visualize(epoch, lift_force_visualize)

        # Joint LBFGS training loop
        if (u_epochs_LBFGS or ϕ_epochs_LBFGS) != 0:
            def closure_joint():
                if u_epochs_LBFGS > 0: u_λ_optimizer_scheduler.optimizer_lbfgs.zero_grad()
                if ϕ_epochs_LBFGS > 0: u_λ_optimizer_scheduler.optimizer_lbfgs.zero_grad()
                ℒ, _, _, _, _ = total_loss(J_global=1, Σxy_global=1, Σyy_global=1, mass_global=1, symmetry_global=1, data_global=1)
                ℒ.backward()
                return ℒ
            
            most_epochs_LBFGS = max(u_epochs_LBFGS, ϕ_epochs_LBFGS) + 1
            for epoch in range(most_epochs_LBFGS):
                if epoch < u_epochs_LBFGS: u_λ_optimizer_scheduler.optimizer_lbfgs.step(closure_joint)
                if epoch < ϕ_epochs_LBFGS: ϕ_λ_optimizer_scheduler.optimizer_lbfgs.step(closure_joint)
                ℒ, ℒ_visualize, ℒ_individuals, ℒ_individuals_visualize, lift_force_visualize = total_loss(J_global=1, Σxy_global=1, Σyy_global=1, mass_global=1, symmetry_global=1, data_global=1)
                if use_scheduler:
                    if epoch < u_epochs_LBFGS: u_λ_optimizer_scheduler.scheduler_lbfgs.step(ℒ.item())
                    if epoch < ϕ_epochs_LBFGS: ϕ_λ_optimizer_scheduler.scheduler_lbfgs.step(ℒ.item())

                # Update loss history for plotting
                loss_history.append(ℒ_visualize, ℒ_individuals_visualize, β)

                # Visualize
                extra = f" | β {β.item():.3f}" if c.β_learnable else ""
                print(f"joint-LBFGS ep {epoch} | ℒ {ℒ_visualize.item():.6g} | u_lbfgs_lr {current_lr(u_PINN_optimizer_scheduler.optimizer_lbfgs):.2e} | ϕ_lbfgs_lr {current_lr(ϕ_PINN_optimizer_scheduler.optimizer_lbfgs):.2e}{extra}")
                print("dp*:", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
                if epoch % visualize_step == 0: visualize(epoch, lift_force_visualize)

# ---------------------------------------------------------------
# One PINN
# ---------------------------------------------------------------
else:
    # Define parameters
    PINN_parameters = list(PINN.parameters()) + [dpstar_dxstar]
    if use_spatially_adaptive_learnable_parameters:
        λ_parameters = [λ_J, λ_Σxy, λ_Σyy, λ_mass, λ_symmetry, λ_data]
    if c.β_learnable: β_parameter = [β]

    # Define optimizers & schedulers 
    PINN_optimizer_scheduler = make_optimizer_scheduler(PINN_parameters, c.single_PINN_learning_rate)
    if use_spatially_adaptive_learnable_parameters: λ_optimizer_scheduler = make_optimizer_scheduler(λ_parameters, c.single_PINN_λ_learning_rate)
    if c.β_learnable: β_optimizer_scheduler = make_optimizer_scheduler(β_parameter, c.β_learning_rate)

    # Histories
    loss_history = LossHistory()
    visualization_history = []

    # Single PINN ADAM training loop
    if single_epochs_ADAM != 0:
        for epoch in range(single_epochs_ADAM + 1):
            PINN_optimizer_scheduler.optimizer_adam.zero_grad()
            if use_spatially_adaptive_learnable_parameters: λ_optimizer_scheduler.optimizer_adam.zero_grad()
            if c.β_learnable: β_optimizer_scheduler.optimizer_adam.zero_grad()

            # Forward & Backward passes
            ℒ, ℒ_visualize, ℒ_individuals, ℒ_individuals_visualize, lift_force_visualize = total_loss(J_global=1, Σxy_global=1, Σyy_global=1, mass_global=1, symmetry_global=1, data_global=1)
            ℒ.backward()

            # Gradient descent & scheduler update
            PINN_optimizer_scheduler.optimizer_adam.step()
            if use_scheduler: PINN_optimizer_scheduler.scheduler_adam.step(ℒ.item())
            if c.β_learnable:
                β_optimizer_scheduler.optimizer_adam.step()
                if use_scheduler: β_optimizer_scheduler.scheduler_adam.step(ℒ.item())

            # Gradient ascent & scheduler update
            if use_spatially_adaptive_learnable_parameters: 
                for λ in λ_parameters:
                    if λ.grad is not None:
                        λ.grad = -λ.grad
                λ_optimizer_scheduler.optimizer_adam.step() 
                if use_scheduler: λ_optimizer_scheduler.scheduler_adam.step(ℒ.item())

            # Update loss history for plotting
            loss_history.append(ℒ_visualize, ℒ_individuals_visualize, β)

            # Visualize
            extra = f" | β {β.item():.3f}" if c.β_learnable else ""
            print(f"single-ADAM ep {epoch} | ℒ {ℒ_visualize.item():.6g} | lr {current_lr(PINN_optimizer_scheduler.optimizer_adam):.2e}{extra}")
            print("dp*:", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
            if epoch % visualize_step == 0: visualize(epoch, lift_force_visualize)

    # Single PINN LBFGS training loop
    if single_epochs_LBFGS != 0:
        def closure_single():
            PINN_optimizer_scheduler.optimizer_lbfgs.zero_grad()
            ℒ, _, _, _, _ = total_loss(J_global=1, Σxy_global=1, Σyy_global=1, mass_global=1, symmetry_global=1, data_global=1)
            ℒ.backward()
            return ℒ
        
        for epoch in range(single_epochs_LBFGS + 1):
            PINN_optimizer_scheduler.optimizer_lbfgs.step(closure_single)
            ℒ, ℒ_visualize, ℒ_individuals, ℒ_individuals_visualize, lift_force_visualize = total_loss(J_global=1, Σxy_global=1, Σyy_global=1, mass_global=1, symmetry_global=1, data_global=1)
            if use_scheduler: PINN_optimizer_scheduler.scheduler_lbfgs.step(ℒ.item())

            # Update loss history for plotting
            loss_history.append(ℒ_visualize, ℒ_individuals_visualize, β)

            # Visualize
            extra = f" | β {β.item():.3f}" if c.β_learnable else ""
            print(f"single-LBFGS ep {epoch} | ℒ {ℒ_visualize.item():.6g} | lr {current_lr(PINN_optimizer_scheduler.optimizer_lbfgs):.2e}{extra}")
            print("dp*:", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
            if epoch % visualize_step == 0: visualize(epoch, lift_force_visualize)

'''# Save model when finished
save_path = c.save_path / 'saved_model'
Path(save_path).mkdir(parents=True, exist_ok=True)
torch.save(ϕ_PINN.state_dict(), save_path)
if c.save_gif: animate_mp4()'''