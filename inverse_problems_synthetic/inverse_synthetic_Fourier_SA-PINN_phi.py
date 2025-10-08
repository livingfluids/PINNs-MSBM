import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------------- 
# SOLVES THE INVERSE PROBLEM FOR ϕ FOR SYNTHETIC DATA USING A FOURIER
# EXPANSION LAYER
# ----------------------------------------------------------------------------- 

# Controls & Hyperparameters --------------------------------------------------
# Device selection: MPS (Mac GPU) > CUDA (NVIDIA GPU) > CPU
USE_GPU = True  # Set to False to force CPU usage

if USE_GPU:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
print(device)
torch.manual_seed(0)
np.random.seed(0)

# Model controls
data_file_1 = True  # which example data file do you want to try? one or two? if one, one = True. if two, one = False
save_images = False  # save images during training for making an animation later
use_scheduler = True  # if True, the learning rates will decrease over time as the loss plateaus

# Define the paths for data, save, and visualization
base_path = Path(__file__).parent
data_path = base_path / "synthetic_data"
save_path = base_path / "saved_models"
visualization_path = base_path / "saved_visuals"

# Define the model names and data file
Ux_model_saved_name = "synthetic_saved_Ux_model_example_1" if data_file_1 else "synthetic_saved_Ux_model_example_2"
ϕ_model_saved_name = "synthetic_saved_phi_example_1" if data_file_1 else "synthetic_saved_phi_example_2"
data_file_name = "synthetic_data_example_1.csv" if data_file_1 else "synthetic_data_example_2.csv"

# Load the data
df = pd.read_csv(data_path / data_file_name)

# PINN parameters 
Ux_NEURONS = 64
ϕ_NEURONS = 64
Ux_LAYERS = 3
ϕ_LAYERS = 3
ϕ_FOURIER_SCALE = 5.0
EPOCHS_ADAM = 100
EPOCHS_LBFGS = 200
ϕ_LEARNING_RATE = 1e-3
λ_LEARNING_RATE = 1e-6  # ideally no larger than the smallest λ_..._INIT value 
COLLOCATION_PTS = 500  # Collocation points

# Scheduler parameters
SCHEDULER_PATIENCE = 50  # Epochs to wait for loss improvement before reducing LR
SCHEDULER_FACTOR = 0.90  # Factor to multiply LR by when reducing (e.g., 0.5 halves it)
SCHEDULER_MIN_LR = 1e-6  # Minimum LR to stop reducing below this

# Initial adaptive weights
λ_J_INIT = 1
λ_Σxy_INIT = 1
λ_Σyy_INIT = 1
λ_mass_INIT = 1
λ_symmetry_INIT = 1

# Physical parameters (match OpenFOAM)
p = torch.tensor(df['p'].values.mean(), device=device, dtype=torch.float32)  # steady state pressure (Pa)
Ux_max = torch.tensor(df['U_0'].values.max(), device=device, dtype=torch.float32)  # max steady state velocity (m/s)
ϕ_average = torch.tensor(df['c'].values.mean(), device=device, dtype=torch.float32)  # average ϕ (dimensionless)
H = torch.tensor(25e-6 if data_file_1 else 50e-6, device=device, dtype=torch.float32)  # channel height (m)
ρ = torch.tensor(1190.0, device=device, dtype=torch.float32)  # solvent density (Kg/m³)
η = torch.tensor(0.48, device=device, dtype=torch.float32)  # dynamic viscosity (Pa·s)
η0 = η / ρ  # kinematic viscosity (m²/s)  
Kn = torch.tensor(0.75, device=device, dtype=torch.float32)  # fitting parameter (dimensionless)
λ2 = torch.tensor(0.8, device=device, dtype=torch.float32)  # fitting parameter (dimensionless)
λ3 = torch.tensor(0.5, device=device, dtype=torch.float32)  # fitting parameter (dimensionless)
α = torch.tensor(4.0, device=device, dtype=torch.float32)  # fitting parameter α ∈ [2, 5] (dimensionless)
a = torch.tensor(2.82e-6, device=device, dtype=torch.float32)  # particle radius (m)
ϕ_max = torch.tensor(0.5, device=device, dtype=torch.float32)  # max ϕ (dimensionless)
ε = a / ((H / 2)**2)  # non-local shear-rate coefficient (1/m)
β = torch.tensor(1.2 if data_file_1 else 1.1, device=device, dtype=torch.float32)  # power-law coefficient 
frv = torch.tensor(1.2, device=device, dtype=torch.float32)  # function of the reduced volume

# Learned parameters
dpstar_dxstar = nn.Parameter(torch.tensor([-2.5], dtype=torch.float32, device=device))  # normalized x-pressure gradient (dimensionless)

# Data tensors
y_data = 2.0 * torch.tensor(df['y'].values, dtype=torch.float32, device=device).unsqueeze(1) / H - 1.0
Ux_data = torch.tensor(df['U_0'].values, dtype=torch.float32, device=device).unsqueeze(1) / Ux_max
ϕ_data = torch.tensor(df['c'].values, dtype=torch.float32, device=device).unsqueeze(1)

# Collocation points (uniform, non-random points required for self-adaptive weights)
y_uniform = torch.linspace(-1.0, 1.0, COLLOCATION_PTS, device=device).unsqueeze(1).requires_grad_(True)

# Trial functions
Ux_trial = lambda y: Ux_PINN(torch.cat([y.to(device)], dim=1))[:,0:1] * (1 + y.to(device)) * (1 - y.to(device))  # torch.Size([y, 1]), normalized 
ϕ_trial = lambda y: ϕ_max * torch.sigmoid(ϕ_PINN(y.to(device))[:,0:1]) * (1 + y.to(device)) * (1 - y.to(device))  # torch.Size([y, 1]), sigmoid keeps bounded between 0 and ϕ_max

# Loss history for plotting
class LossHistory:
    def __init__(self):
        self.total = []
        self.individuals = []

    def append(self, ℒ, ℒ_individuals):
        self.total.append(ℒ.item())
        self.individuals.append([ℒ_individual.item() for ℒ_individual in ℒ_individuals])

# PINN ------------------------------------------------------------------------
# Fourier features from Tancik et al. 
class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size, scale):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
# 'Jagtap' activation function from Jagtap et al. improves convergence by an insignificant amount, so this isn't necessary
class JagtapActivation(nn.Module):
    def __init__(self, neurons):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, neurons))
        self.f = nn.Parameter(torch.randn(1, neurons))
        self.tanh = nn.Tanh()

    def forward(self, x):
        activation_function = self.alpha * self.tanh(self.f * x)
        return activation_function

# Compile Ux_PINN layers
layers = [nn.Linear(1, Ux_NEURONS), nn.Tanh()]
for layer in range(Ux_LAYERS): layers.extend([nn.Linear(Ux_NEURONS, Ux_NEURONS), nn.Tanh()])
layers.append(nn.Linear(Ux_NEURONS, 1))

# Create the Ux_PINN
Ux_PINN = nn.Sequential(*layers).to(device)

# Compile ϕ_PINN layers 
layers = [FourierFeatures(in_features=1, mapping_size=ϕ_NEURONS, scale=ϕ_FOURIER_SCALE), nn.Linear(2 * ϕ_NEURONS, ϕ_NEURONS), JagtapActivation(neurons=ϕ_NEURONS)]
for layer in range(ϕ_LAYERS): layers.extend([nn.Linear(ϕ_NEURONS, ϕ_NEURONS), JagtapActivation(neurons=ϕ_NEURONS)])
layers.append(nn.Linear(ϕ_NEURONS, 1))

# Create the ϕ_PINN
ϕ_PINN = nn.Sequential(*layers).to(device)

# Global Weights (Will Better Utilize Later) ----------------------------------
m_J = 1
m_Σxy = 1
m_Σyy = 1
m_mass = 1
m_symmetry = 1

# Self-Adaptive Weights -------------------------------------------------------
# Self-Adaptive Weights from McClenny & Braga-Neto 
mask = lambda λ: λ**2  # torch.nn.functional.softplus(λ) can also be tried

# Self-adaptive weights, all start off as ones, but NOT all are the same tensor shape
λ_J = nn.Parameter(torch.ones([COLLOCATION_PTS, 1], device=device, requires_grad=True) * λ_J_INIT)  # for Jstar_divergence, a vector
λ_Σxy = nn.Parameter(torch.ones([COLLOCATION_PTS, 1], device=device, requires_grad=True) * λ_Σxy_INIT)  # for dΣxystar_dystar, a vector
λ_Σyy = nn.Parameter(torch.ones([COLLOCATION_PTS, 1], device=device, requires_grad=True) * λ_Σyy_INIT)  # for dΣyystar_dystar, a vector
λ_mass = nn.Parameter(torch.ones([1], device=device, requires_grad=True) * λ_mass_INIT)  # for mass conservation, a scalar
λ_symmetry = nn.Parameter(torch.ones([1], device=device, requires_grad=True) * λ_symmetry_INIT)  # for symmetry, a scalar 

# Loss Functions --------------------------------------------------------------
def physics_loss():  # physics ensures ∇⋅J = ∇⋅Σ = 0
    ystar = y_uniform  # already normalized 
    Uxstar = Ux_trial(ystar)  # already normalized 
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
    dUxstar_dystar = torch.autograd.grad(Uxstar, ystar, torch.ones_like(Uxstar), create_graph=True)[0]  # torch.Size([y, 1])
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
    left_wall = 3 * η0 * γ̇ / (4 * torch.pi * ((H/2)*(ystar + 1) + a)**β) * frv
    right_wall = 3 * η0 * γ̇ / (4 * torch.pi * ((H/2)*(1 - ystar) + a)**β) * frv
    scale_L = (H ** 2) / (2 * η0 * Ux_max)  # nondimensionalize after calculating it
    L = torch.stack([
        torch.cat([zero], dim=1),
        torch.cat([scale_L * (left_wall - right_wall)], dim=1),
        torch.cat([zero], dim=1)
    ], dim=1)
    
    # Diagonal tensor of the SBM (Q)
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, λ2, 0.0], [0.0, 0.0, λ3]], device=device).repeat(ystar.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Non-local shear rate tensor
    γ̇NLstar = ε * H / 2

    # Particle normal stress diagonal tensor (Σₙₙᵖ)
    Σpnnstar = ηN(ϕ).view(-1, 1, 1) * (γ̇star.unsqueeze(1) + γ̇NLstar) * Q  # torch.Size([y, 3, 3]), a matrix for each y

    # Oriented particle stress tensor (Σᵖ)
    Σpstar = -Σpnnstar + (2 * ηp(ϕ).view(-1, 1, 1) * Estar)  # torch.Size([y, 3, 3]), a matrix for each y

    # Divergence of oriented particle stress tensor (∇⋅Σᵖ)
    dΣpxystar_dystar = torch.autograd.grad(Σpstar[:, 0, 1], ystar, torch.ones_like(Σpstar[:, 0, 1]), create_graph=True)[0]  # torch.Size([y, 1])
    dΣpyystar_dystar = torch.autograd.grad(Σpstar[:, 1, 1], ystar, torch.ones_like(Σpstar[:, 1, 1]), create_graph=True)[0]  # torch.Size([y, 1])
    Σpstar_divergence = torch.stack([
        torch.cat([zero + dΣpxystar_dystar + zero], dim=1),
        torch.cat([zero + dΣpyystar_dystar + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y

    # Migration flux (J)
    Jstar = - (2 * A**2 / 9) * f(ϕ).unsqueeze(1) * (Σpstar_divergence + ϕ.view(-1, 1, 1) * L)  # torch.Size([y, 3, 1])

    # Divergence of migration flux (∇⋅J)
    dJxstar_dxstar = dJzstar_dzstar = zero
    dJystar_dystar = torch.autograd.grad(Jstar[:, 1, 0], ystar, torch.ones_like(Jstar[:, 1, 0]), create_graph=True)[0]  # torch.Size([y, 1])
    Jstar_divergence = dJxstar_dxstar + dJystar_dystar + dJzstar_dzstar  # torch.Size([y, 1])

    # Identity matrix (I)
    I = torch.eye(3, device=device).repeat(ystar.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid phase stress (Σᶠ)
    Σfstar = - pstar * I + 2 * Estar

    # Total stress (Σ)
    Σstar = Σpstar + Σfstar

    # Suspension momentum balance (∇⋅Σ)
    dΣxystar_dystar = torch.autograd.grad(Σstar[:, 0, 1], ystar, torch.ones_like(Σstar[:, 0, 1]), create_graph=True)[0] - dpstar_dxstar
    dΣyystar_dystar = torch.autograd.grad(Σstar[:, 1, 1], ystar, torch.ones_like(Σstar[:, 1, 1]), create_graph=True)[0]

    return Jstar_divergence, dΣxystar_dystar, dΣyystar_dystar

def ϕ_bulk_loss():  # IC ensures mean(ϕ) never changes
    ystar = y_uniform   # already normalized
    ϕ = ϕ_trial(ystar)

    # Mass conservation error calculation
    ϕ_bulk_term = torch.mean(ϕ) - ϕ_average 

    return ϕ_bulk_term

def ϕ_symmetry_loss():  # ensures ϕ is symmetric along centerflow axis
    ystar = y_uniform   # already normalized

    # Symmetry error calculation 
    ϕ_symmetry_term = ϕ_trial(ystar) - ϕ_trial(-ystar)

    return ϕ_symmetry_term

def total_loss():  # combining losses
    Jstar_divergence, dΣxystar_dystar, dΣyystar_dystar = physics_loss()  # torch.Size([y, 1]) for each
    ϕ_bulk_term = ϕ_bulk_loss()  # torch.Size([1])
    ϕ_symmetry_term = ϕ_symmetry_loss()  # torch.Size([1])

    # Indivisual losses, in the style of McClenny & Braga-Neto, all become scalars
    ℒ_J = m_J * torch.sum(mask(λ_J) * Jstar_divergence**2)
    ℒ_Σxy = m_Σxy * torch.sum(mask(λ_Σxy) * dΣxystar_dystar**2)
    ℒ_Σyy = m_Σyy * torch.sum(mask(λ_Σyy) * dΣyystar_dystar**2)
    ℒ_mass = m_mass * torch.sum(mask(λ_mass) * ϕ_bulk_term**2)
    ℒ_symmetry = m_symmetry * torch.sum(mask(λ_symmetry) * ϕ_symmetry_term**2)
    # NOTE ℒ_mass does not need mean(...), as it is already scalars, but for the the sake of code consistiency, they are

    # loss function, in the style of McClenny & Braga-Neto
    ℒ = torch.sqrt(ℒ_J) + torch.sqrt(ℒ_Σxy) + torch.sqrt(ℒ_Σyy) + torch.sqrt(ℒ_mass) + torch.sqrt(ℒ_symmetry)  # sqrt helps converge faster, from Urbán et al.

    # Unweighted losses for visualization
    ℒ_J_un = torch.sum(torch.sqrt(Jstar_divergence**2))
    ℒ_Σxy_un = torch.sum(torch.sqrt(dΣxystar_dystar**2))
    ℒ_Σyy_un = torch.sum(torch.sqrt(dΣyystar_dystar**2))
    ℒ_mass_un = torch.sum(torch.sqrt(ϕ_bulk_term**2))
    ℒ_symmetry_un = torch.sum(torch.sqrt(ϕ_symmetry_term**2))

    # Unweighted loss for visualization of loss decrease
    ℒ_un = ℒ_J_un + ℒ_Σxy_un + ℒ_Σyy_un + ℒ_mass_un + ℒ_symmetry_un
 
    # Individual losses for tracking and eventual visualizatio for debugging
    ℒ_individuals = [ℒ_J_un, ℒ_Σxy_un, ℒ_Σyy_un, ℒ_mass_un, ℒ_symmetry_un]

    return ℒ, ℒ_un, ℒ_individuals

# Visualize -------------------------------------------------------------------
def visualize(epoch):
    with torch.no_grad():
        y_plot = torch.linspace(-1.0, 1.0, COLLOCATION_PTS, device=device).unsqueeze(1)
        y_plot_dim = ((y_plot + 1.0) / 2.0 * H).cpu().numpy()
        Ux_pinn = (Ux_trial(y_plot) * Ux_max).cpu().numpy()
        phi_pinn = ϕ_trial(y_plot).cpu().numpy()

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        ax_ux = axs[0][0]
        ax_phi = axs[0][1]
        ax_total_loss = axs[1][1]
        ax_indiv_loss = axs[1][0]

        # Ux
        ax_ux.plot(((y_data + 1)/2 * H).cpu(), (Ux_data * Ux_max).cpu(), 'ko', markersize=3, label='Data')
        ax_ux.plot(y_plot_dim, Ux_pinn, 'b-', label='PINN')
        ax_ux.set_xlabel('y [m]')
        ax_ux.set_ylabel('Ux [m/s]')
        ax_ux.legend()
        ax_ux.grid(True)

        # Phi
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
            ax_total_loss.set_ylabel('Loss')
            ax_total_loss.legend()
            ax_total_loss.grid(True)

        # Individual Losses plot
        if loss_history.individuals: # ∇⋅J = ∇⋅Σ
            component_names = ["∇⋅J", "∇⋅Σ (xy)", "∇⋅Σ (yy)", "IC", "sym"]
            for i, indiv_loss in enumerate(zip(*loss_history.individuals)):
                ax_indiv_loss.semilogy(indiv_loss, label=f'Loss {component_names[i]}')
            ax_indiv_loss.set_xlabel('Epoch')
            ax_indiv_loss.set_ylabel('Loss')
            ax_indiv_loss.legend()
            ax_indiv_loss.grid(True)

        plt.tight_layout()
        plt.show(block=False)
        if save_images:
            plt.savefig(visualization_path / f'plot_epoch_{epoch}.png')
        plt.pause(0.1)
        plt.close(fig)

# Training Loop ---------------------------------------------------------------
# Define parameters
ϕ_PINN_parameters = list(ϕ_PINN.parameters()) + [dpstar_dxstar]  # PINN parameters (include all)
λ_parameters = [λ_J, λ_Σxy, λ_Σyy, λ_mass, λ_symmetry]  # self-adaptive weights (include all)

# Define optimizers
ϕ_PINN_optimizer_Adam = torch.optim.Adam(ϕ_PINN_parameters, lr=ϕ_LEARNING_RATE)
ϕ_PINN_optimizer_LBFGS = torch.optim.LBFGS(ϕ_PINN_parameters, lr=ϕ_LEARNING_RATE)
λ_optimizer = torch.optim.Adam(λ_parameters, lr=λ_LEARNING_RATE)

# Define schedulers (ReduceLROnPlateau type does exactly that)
ϕ_PINN_scheduler_Adam = torch.optim.lr_scheduler.ReduceLROnPlateau(ϕ_PINN_optimizer_Adam, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
ϕ_PINN_scheduler_LBFGS = torch.optim.lr_scheduler.ReduceLROnPlateau(ϕ_PINN_optimizer_LBFGS, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
λ_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(λ_optimizer, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

# Call the loss history class
loss_history = LossHistory()

# Load previously saved Ux model
Ux_saved_path = save_path / Ux_model_saved_name
Ux_PINN.load_state_dict(torch.load(f=Ux_saved_path))
Ux_PINN.eval()

# ADAM training loop 
for epoch in range(EPOCHS_ADAM):
    ϕ_PINN_optimizer_Adam.zero_grad() 
    λ_optimizer.zero_grad()

    # Forward & Backward passes
    ℒ, ℒ_un, ℒ_individuals = total_loss()  # forward pass to compute loss
    ℒ.backward()  # backward pass to compute gradients

    # Gradient descent & scheduler update
    ϕ_PINN_optimizer_Adam.step()  # gradient descent updating PINN parameters
    ϕ_PINN_scheduler_Adam.step(ℒ_un.item()) if use_scheduler else None

    # Make gradient negative for λ parameters to achieve ascent, rather than descent
    for λ in λ_parameters:
        if λ.grad is not None:
            λ.grad = -λ.grad

    # Gradient ascent & scheduler update
    λ_optimizer.step()  # gradient ascent updating self-adaptive weights
    λ_scheduler.step(ℒ_un.item()) if use_scheduler else None

    # Update loss history for plotting
    loss_history.append(ℒ_un, ℒ_individuals)

    # Visuals
    if use_scheduler:
        print(f"Epoch: {epoch} | Loss: {ℒ_un.item()} | Individual Losses: {[f'{l.item():.5f}' for l in ℒ_individuals]} | ϕ Lr: {ϕ_PINN_scheduler_LBFGS.get_last_lr()} | λ Lr: {λ_scheduler.get_last_lr()}")
    else:
        print(f"Epoch: {epoch} | Loss: {ℒ_un.item()} | Individual Losses: {[f'{l.item():.5f}' for l in ℒ_individuals]}")
    print("Normalized and non-normalized pressure gradient: ", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
    if epoch % 50 == 0:
        visualize(epoch)

# LBFGS closure function
def closure():
    ϕ_PINN_optimizer_LBFGS.zero_grad()
    ℒ, ℒ_un, ℒ_individuals = total_loss()
    ℒ.backward()
    return ℒ

# LBFGS training loop 
for epoch in range(EPOCHS_LBFGS):
    ϕ_PINN_optimizer_LBFGS.step(closure)
    ℒ, ℒ_un, ℒ_individuals = total_loss()
    loss_history.append(ℒ_un, ℒ_individuals)
    ϕ_PINN_scheduler_LBFGS.step(ℒ_un.item()) if use_scheduler else None

    # Visuals
    if use_scheduler:
        print(f"Epoch: {epoch} | Loss: {ℒ_un.item()} | Individual Losses: {[f'{l.item():.5f}' for l in ℒ_individuals]} | ϕ Lr: {ϕ_PINN_scheduler_LBFGS.get_last_lr()} | λ Lr: {λ_scheduler.get_last_lr()}")
    else:
        print(f"Epoch: {epoch} | Loss: {ℒ_un.item()} | Individual Losses: {[f'{l.item():.5f}' for l in ℒ_individuals]}")
    print("Normalized and non-normalized pressure gradient: ", dpstar_dxstar.item(), (dpstar_dxstar * (4 * η0 * Ux_max) / (H ** 2)).item())
    if epoch % 10 == 0:
        visualize(epoch)

# Save ϕ model when finished
ϕ_saved_path = save_path / ϕ_model_saved_name
Path(save_path).mkdir(parents=True, exist_ok=True)
torch.save(ϕ_PINN.state_dict(), ϕ_saved_path)
