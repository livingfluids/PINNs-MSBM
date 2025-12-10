import torch
from torch import nn
import paths
from dataclasses import dataclass
from typing import Union
import config

df = paths.df
params = paths.params

@dataclass
class Params:
    u_max: torch.Tensor
    y_data_: torch.Tensor
    y_coll_: torch.Tensor
    u_data_: torch.Tensor
    ϕ_data:torch.Tensor
    ϕ_max: torch.Tensor
    ϕ_bulk: torch.Tensor
    H: torch.Tensor
    ρ: torch.Tensor
    η: torch.Tensor
    η0: torch.Tensor
    Kn: torch.Tensor
    λ2: torch.Tensor
    λ3: torch.Tensor
    α: torch.Tensor
    β: Union[nn.Parameter, torch.Tensor]
    a: torch.Tensor
    ε: torch.Tensor
    frv: torch.Tensor
    p: torch.Tensor
    dp_dx: nn.Parameter
    H0: torch.Tensor
    mask: callable
    λ_J: torch.Tensor
    λ_Σxy: torch.Tensor
    λ_Σyy: torch.Tensor
    λ_mass: torch.Tensor
    λ_symmetry: torch.Tensor
    λ_data: torch.Tensor
    T_max: torch.Tensor
    exp: torch.Tensor

# Placeholders
device = torch.device('cpu')
dtype = torch.float32

def generate_params() -> Params:
    # Scalars from CSV
    u_max = torch.tensor(df["u"].values.max(), device=device, dtype=dtype)  # max u                       (m/s)

    # Scalars from YAML
    ϕ_max   = torch.tensor(params["phi_max"],  device=device, dtype=dtype)  # max ϕ                       (dimensionless)
    ϕ_bulk  = torch.tensor(params["phi_bulk"], device=device, dtype=dtype)  # bulk ϕ                      (dimensionless)
    H       = torch.tensor(params["H"],        device=device, dtype=dtype)  # channel height              (m)
    ρ       = torch.tensor(params["rho"],      device=device, dtype=dtype)  # solvent density             (kg/m³)
    η       = torch.tensor(params["eta"],      device=device, dtype=dtype)  # dynamic viscosity           (Pa·s)
    Kn      = torch.tensor(params["Kn"],       device=device, dtype=dtype)  # fitting parameter           (dimensionless)
    λ2      = torch.tensor(params["lambda2"],  device=device, dtype=dtype)  # fitting parameter           (dimensionless)
    λ3      = torch.tensor(params["lambda3"],  device=device, dtype=dtype)  # fitting parameter           (dimensionless)
    α       = torch.tensor(params["alpha"],    device=device, dtype=dtype)  # α ∈ [2, 5]                  ()
    a       = torch.tensor(params["a"],        device=device, dtype=dtype)  # particle radius             (m)  
    frv     = torch.tensor(params["frv"],      device=device, dtype=dtype)  # function of reduced volume  ()
    p       = torch.tensor(params["p"],        device=device, dtype=dtype)  # pressure                    ()
    H0      = torch.tensor(params["H0"],       device=device, dtype=dtype)  # buffer                      ()
    β       = torch.tensor(params["beta"],     device=device, dtype=dtype)  # lift force exponent         ()

    # Derived quantities
    η0      = η / ρ                 # kinematic viscosity         (m²/s)       
    ε       = a / ((H / 2.0) ** 2)  # non-local shear-rate coeff. (1/m)

    # Learnable quantities
    dp_dx   = nn.Parameter(torch.tensor([config.dp_dx_INIT], device=device, dtype=dtype, requires_grad=True))  # pressure gradient ()
    exp     = nn.Parameter(torch.tensor([2.0], device=device, dtype=dtype, requires_grad=True))

    # Tensors from CSV
    y_data_ = 2.0 * torch.tensor(df['y'].values, dtype=torch.float32, device=device).unsqueeze(1) / H - 1.0
    y_coll_ = torch.linspace(-1.0, 1.0, config.COLL, device=device).unsqueeze(1).requires_grad_(True)
    u_data_ = torch.tensor(df['u'].values, dtype=torch.float32, device=device).unsqueeze(1) / u_max
    ϕ_data  = torch.tensor(df['phi'].values, dtype=torch.float32, device=device).unsqueeze(1)

    # Spatially Self-Adaptive Weights (McClenny & Braga-Neto)
    mask        = lambda λ: λ**2
    λ_J         = nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype, requires_grad=True))
    λ_Σxy       = nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype, requires_grad=True))
    λ_Σyy       = nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype, requires_grad=True))
    λ_mass      = nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype, requires_grad=True))
    λ_symmetry  = nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype, requires_grad=True))
    λ_data      = nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype, requires_grad=True))

    # Scheduler parameter
    T_max       = nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype, requires_grad=True))

    return Params(
            u_max=u_max,
            y_data_=y_data_,
            y_coll_=y_coll_,
            u_data_=u_data_,
            ϕ_data=ϕ_data,
            ϕ_max=ϕ_max,
            ϕ_bulk=ϕ_bulk,
            H=H,
            ρ=ρ,
            η=η,
            η0=η0,
            Kn=Kn,
            λ2=λ2,
            λ3=λ3,
            α=α,
            β=β,
            a=a,
            ε=ε,
            frv=frv,
            p=p,
            dp_dx=dp_dx,
            H0=H0,
            mask=mask,
            λ_J=λ_J,
            λ_Σxy=λ_Σxy,
            λ_Σyy=λ_Σyy,
            λ_mass=λ_mass,
            λ_symmetry=λ_symmetry,
            λ_data=λ_data,
            T_max=T_max,
            exp=exp,
        )