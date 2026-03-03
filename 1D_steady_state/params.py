import torch
from torch import nn
import paths
from dataclasses import dataclass
from typing import Union
import config
import pandas as pd
import yaml

# Parameter Data Class
@dataclass
class Params:
    u_max: torch.Tensor
    ϕ_max: torch.Tensor
    ϕ_bulk: torch.Tensor
    # - 
    y_data_: torch.Tensor
    y_coll_: torch.Tensor
    # - 
    u_data_: torch.Tensor
    ϕ_data_:torch.Tensor
    # - 
    H: torch.Tensor
    ρ: torch.Tensor
    η: torch.Tensor
    η0: torch.Tensor
    Kn: torch.Tensor
    λ2: torch.Tensor
    λ3: torch.Tensor
    α: torch.Tensor
    a: torch.Tensor
    # - 
    β: Union[nn.Parameter, torch.Tensor]
    β_true: Union[torch.Tensor, None]
    # - 
    ε: torch.Tensor
    frv: torch.Tensor
    p: torch.Tensor
    H0: torch.Tensor
    # - 
    mask: callable
    λ_J: nn.Parameter
    λ_Jy_wall: nn.Parameter
    λ_Σxy: nn.Parameter
    λ_Σyy: nn.Parameter
    λ_mass: nn.Parameter
    λ_BC: nn.Parameter
    λ_u_sym: nn.Parameter
    λ_ϕ_sym: nn.Parameter
    λ_data: nn.Parameter
    # - 
    Λ_PDEs: nn.Parameter
    Λ_BCs: nn.Parameter
    Λ_data: nn.Parameter
    # - 
    dp_dx_: nn.Parameter
    CFL_: torch.Tensor

# Build Parameters 
def buildParams(device) -> Params:
    dtype       = torch.float32

    # Read Data
    df          = pd.read_csv(paths.data_dir / 'data.csv')
    with open(paths.data_dir / 'parameters.yaml', 'r') as file: params = yaml.safe_load(file)

    # Scalars from CSV
    u_max       = torch.tensor(df["u"].values.max(), device=device, dtype=dtype)  # max u                       (m/s)

    # Scalars from YAML
    ϕ_max       = torch.tensor(params["phi_max"],  device=device, dtype=dtype)  # max ϕ                       (dimensionless)
    ϕ_bulk      = torch.tensor(params["phi_bulk"], device=device, dtype=dtype)  # bulk ϕ                      (dimensionless)
    H           = torch.tensor(params["H"],        device=device, dtype=dtype)  # channel height              (m)
    ρ           = torch.tensor(params["rho"],      device=device, dtype=dtype)  # solvent density             (kg/m³)
    η           = torch.tensor(params["eta"],      device=device, dtype=dtype)  # dynamic viscosity           (Pa·s)
    Kn          = torch.tensor(params["Kn"],       device=device, dtype=dtype)  # fitting parameter           (dimensionless)
    λ2          = torch.tensor(params["lambda2"],  device=device, dtype=dtype)  # fitting parameter           (dimensionless)
    λ3          = torch.tensor(params["lambda3"],  device=device, dtype=dtype)  # fitting parameter           (dimensionless)
    α           = torch.tensor(params["alpha"],    device=device, dtype=dtype)  # α ∈ [2, 5]                  ()
    a           = torch.tensor(params["a"],        device=device, dtype=dtype)  # particle radius             (m)  
    frv         = torch.tensor(params["frv"],      device=device, dtype=dtype)  # function of reduced volume  ()
    p           = torch.tensor(params["p"],        device=device, dtype=dtype)  # pressure                    ()
    H0          = torch.tensor(params["H0"],       device=device, dtype=dtype)  # buffer                      ()
    dp_dx       = torch.tensor(params["drho_dx"],  device=device, dtype=dtype)  # pressure gradient           (Pa/m)
    CFL         = torch.tensor(params["CFL"],      device=device, dtype=dtype)  # CFL                         (m)
    β_true      = torch.tensor(params["beta"],    device=device, dtype=dtype)   # lift force exponent         ()

    # Derived quantities
    η0          = η                                                             # kinematic viscosity         (m²/s)       
    ε           = a / ((H / 2.0) ** 2)                                          # non-local shear-rate coeff. (1/m)
    dp_dx_      = dp_dx * H**2 / (4 * η0 * u_max)                               # normalized dp_dx            (dimensionless)
    CFL_        = int(CFL / H * (config.COLL - 1))                              # normalized CFL              (dimensionless)

    # Tensors from CSV
    y_data_     = 2.0 * torch.tensor(df['y'].values, dtype=dtype, device=device).unsqueeze(1) / H - 1.0
    y_coll_     = torch.linspace(-1.0, 1.0, config.COLL, device=device).unsqueeze(1).requires_grad_(True)
    u_data_     = torch.tensor(df['u'].values, dtype=dtype, device=device).unsqueeze(1) / u_max
    ϕ_data_     = torch.tensor(df['phi'].values, dtype=dtype, device=device).unsqueeze(1)

    # Spatially Self-Adaptive Weights (McClenny & Braga-Neto)
    mask        = config.λ_MASK
    λ_J         = nn.Parameter(torch.ones([config.COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_Jy_wall   = nn.Parameter(torch.ones([2], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_Σxy       = nn.Parameter(torch.ones([config.COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_Σyy       = nn.Parameter(torch.ones([config.COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_mass      = nn.Parameter(torch.ones([1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_BC        = nn.Parameter(torch.ones([CFL_, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_u_sym     = nn.Parameter(torch.ones([config.COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_ϕ_sym     = nn.Parameter(torch.ones([config.COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_data      = nn.Parameter(torch.ones([y_data_.size()[0], 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)

    # Globally Self-Adaptive Weights
    Λ_PDEs      = nn.Parameter(torch.ones([1], device=device, dtype=dtype, requires_grad=True))
    Λ_BCs       = nn.Parameter(torch.ones([1], device=device, dtype=dtype, requires_grad=True))
    Λ_data      = nn.Parameter(torch.ones([1], device=device, dtype=dtype, requires_grad=True))

    # Beta Handling 
    if config.CASE == 'learn beta': β = nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype, requires_grad=True))
    elif config.CASE == 'learn cfl': β = β_true

    return Params(
            u_max=u_max,
            y_data_=y_data_,
            y_coll_=y_coll_,
            u_data_=u_data_,
            ϕ_data_=ϕ_data_,
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
            β_true=β_true,
            a=a,
            ε=ε,
            frv=frv,
            p=p,
            H0=H0,
            mask=mask,
            λ_J=λ_J,
            λ_Jy_wall=λ_Jy_wall,
            λ_Σxy=λ_Σxy,
            λ_Σyy=λ_Σyy,
            λ_mass=λ_mass,
            λ_BC=λ_BC,
            λ_u_sym=λ_u_sym,
            λ_ϕ_sym=λ_ϕ_sym,
            λ_data=λ_data,
            Λ_PDEs=Λ_PDEs,
            Λ_BCs=Λ_BCs,
            Λ_data=Λ_data,
            dp_dx_=dp_dx_,
            CFL_=CFL_,
        )
