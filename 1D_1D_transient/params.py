import torch
from torch import nn
import paths
from dataclasses import dataclass
from typing import Union
import config
import pandas as pd
import yaml
import numpy as np

# Parameter Data Class
@dataclass
class Params:
    u_max: torch.Tensor
    ϕ_max: torch.Tensor
    ϕ_bulk: torch.Tensor
    # - 
    t_data_: torch.Tensor
    t_coll_: torch.Tensor
    y_data_: torch.Tensor
    y_coll_: torch.Tensor
    # - 
    u_data_: torch.Tensor
    ϕ_data_: torch.Tensor
    # 
    y_grid_: torch.Tensor
    t_grid_: torch.Tensor
    # - 
    y_flat_: torch.Tensor
    t_flat_: torch.Tensor
    # -
    t_log_min: torch.Tensor
    t_log_span: torch.Tensor
    t_eps: torch.Tensor
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
    β_true: torch.Tensor
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
    dp_dx_: torch.Tensor
    CFL_: torch.Tensor

# Build Parameters 
def buildParams(device) -> Params:
    dtype = torch.float32
    y_data, t_data, u_data, ϕ_data, p_values, u_maxes = [], [], [], [], [], []

    # Read CSV and YAML
    for data_file in sorted(paths.data_dir.glob('*.csv')):
        df = pd.read_csv(data_file)
        df.columns = df.columns.str.strip()
        # - 
        p_values.append(df['p'].values.mean().item())                                           # [1] per CSV
        u_maxes.append(df['U:0'].values.max().item())                                           # [1] per CSV
        y_data.append(df['arc_length'].values) if y_data == [] else None                        # [y, 1] per CSV
        t_data.append(df.at[0, 'Time'])                                                         # [1] per CSV
        u_data.append(torch.tensor(df['U:0'].values, dtype=dtype, device=device).unsqueeze(1))  # [y, 1] per CSV
        ϕ_data.append(torch.tensor(df['c'].values, dtype=dtype, device=device).unsqueeze(1))    # [y, 1] per CSV
        # - 
    with open(paths.data_dir / 'parameters.yaml', 'r') as file: params = yaml.safe_load(file)

    # Scalars from CSV
    u_max       = torch.tensor(max(u_maxes), device=device, dtype=dtype)        # max u                       (m/s)
    t_min       = min(t_data)                                                   # min t                       (s)
    t_max       = max(t_data)                                                   # max t                       (s)

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
    p           = torch.tensor(np.mean(p_values),      device=device, dtype=dtype)  # pressure                    (Pa)
    H0          = torch.tensor(params["H0"],       device=device, dtype=dtype)  # buffer                      ()
    dp_dx       = torch.tensor(params["drho_dx"],  device=device, dtype=dtype)  # pressure gradient           (Pa/m)
    CFL         = torch.tensor(params["CFL"],      device=device, dtype=dtype)  # CFL                         (m)
    β_true      = torch.tensor(params["beta"],     device=device, dtype=dtype)  # lift force exponent         ()

    # Derived quantities
    η0          = η / ρ                                                         # kinematic viscosity         (m²/s)       
    ε           = a / ((H / 2.0) ** 2)                                          # non-local shear-rate coeff. (1/m)
    dp_dx_      = dp_dx * H**2 / (4 * η0 * u_max)                               # normalized dp_dx            (dimensionless)
    CFL_        = int(CFL.item() / H * (config.y_COLL - 1))                     # normalized CFL              (dimensionless)

    # Tensors from CSV
    t_eps               = torch.tensor(1e-8, device=device, dtype=dtype)                                     # avoid log(0)
    t_min               = torch.tensor(t_min, device=device, dtype=dtype)
    t_max               = torch.tensor(t_max, device=device, dtype=dtype)
    t_log_min           = torch.log(t_min + t_eps)
    t_log_span          = torch.log(t_max + t_eps) - t_log_min
    t_data_tensor       = torch.tensor(t_data, dtype=dtype, device=device).unsqueeze(1)                      # [t, 1] raw
    t_data_log          = (torch.log(t_data_tensor + t_eps) - t_log_min) / t_log_span                       # [t, 1] log-normalized
    t_data_, indices    = torch.sort(t_data_log, dim=0)                                                       # [t, 1] sorted log-normalized
    t_base              = torch.linspace(0.0, 1.0, config.t_COLL, device=device)                             # [t]
    t_coll_phys         = t_min + (t_max - t_min) * t_base.pow(config.T_COLL_EXPONENT)                       # [t] warped in physical time
    t_coll_             = ((torch.log(t_coll_phys + t_eps) - t_log_min) / t_log_span).unsqueeze(1).requires_grad_(True)            # [t, 1]
    # - 
    y_data_             = (2.0 * (torch.tensor(y_data[0], dtype=torch.float32, device=device) / H) - 1.0).unsqueeze(1)      # [y, 1] w.r.t. data
    y_coll_             = torch.linspace(-1.0, 1.0, config.y_COLL, device=device).unsqueeze(1).requires_grad_(True)         # [y, 1] w.r.t. collocations
    # - 
    sorted_idx          = indices.squeeze(-1)
    u_data_             = torch.stack(u_data).index_select(0, sorted_idx) / u_max                                           # [t, y, 1]
    ϕ_data_             = torch.stack(ϕ_data).index_select(0, sorted_idx)                                                   # [t, y, 1]
    # -
    y_grid_, t_grid_    = torch.meshgrid(y_coll_.squeeze(), t_coll_.squeeze(), indexing='ij')                               # [t, y, 1], [t, y, 1]
    y_flat_, t_flat_    = y_grid_.reshape(-1, 1), t_grid_.reshape(-1, 1)                                                    # [y·t, 1], [y·t, 1]

    # Spatially Self-Adaptive Weights (McClenny & Braga-Neto)
    mask        = lambda λ: λ**2
    λ_J         = nn.Parameter(torch.ones([config.t_COLL * config.y_COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_Jy_wall   = nn.Parameter(torch.ones([2, config.t_COLL], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_Σxy       = nn.Parameter(torch.ones([config.t_COLL * config.y_COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_Σyy       = nn.Parameter(torch.ones([config.t_COLL * config.y_COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_mass      = nn.Parameter(torch.ones([config.t_COLL], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_BC        = nn.Parameter(torch.ones([config.y_COLL + 2 * CFL_], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_u_sym     = nn.Parameter(torch.ones([config.t_COLL, config.y_COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_ϕ_sym     = nn.Parameter(torch.ones([config.t_COLL, config.y_COLL, 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)
    λ_data      = nn.Parameter(torch.ones([y_data_.size()[0], t_data_.size()[0], 1], device=device, dtype=dtype, requires_grad=True) * config.λ_INIT)

    # Globally Self-Adaptive Weights
    Λ_PDEs      = nn.Parameter(torch.ones([1], device=device, dtype=dtype, requires_grad=True))
    Λ_BCs       = nn.Parameter(torch.ones([1], device=device, dtype=dtype, requires_grad=True))
    Λ_data      = nn.Parameter(torch.ones([1], device=device, dtype=dtype, requires_grad=True))

    # Learnable quantities
    β           = nn.Parameter(torch.tensor([params["beta"]], device=device, dtype=dtype, requires_grad=True))

    return Params(
            u_max=u_max,
            t_data_=t_data_,
            t_coll_=t_coll_,
            y_data_=y_data_,
            y_coll_=y_coll_,
            u_data_=u_data_,
            ϕ_data_=ϕ_data_,
            y_grid_=y_grid_,
            t_grid_=t_grid_,
            y_flat_=y_flat_,
            t_flat_=t_flat_,
            t_log_min=t_log_min,
            t_log_span=t_log_span,
            t_eps=t_eps,
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
