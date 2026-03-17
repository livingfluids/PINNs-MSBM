import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
import csv
import yaml
import geometry
import config
import pandas as pd
import paths

# Parameter Data Class
@dataclass
class Params:
    R0: torch.Tensor
    R1: torch.Tensor
    R2: torch.Tensor
    # -
    α1: torch.Tensor
    α2: torch.Tensor
    # -
    LEN_PARENT: torch.Tensor
    LEN_BRANCH: torch.Tensor
    # -
    u_data: torch.Tensor
    u_max: torch.Tensor
    ϕ_max: torch.Tensor
    ϕ_bulk: torch.Tensor 
    p_max: torch.Tensor
    # -
    ρ: torch.Tensor
    η: torch.Tensor
    Kn: torch.Tensor
    λ2: torch.Tensor
    λ3: torch.Tensor
    α: torch.Tensor
    a: torch.Tensor
    β: torch.Tensor
    frv: torch.Tensor
    H0: torch.Tensor
    # - 
    S: torch.Tensor
    η0: torch.Tensor
    ε: torch.Tensor
    dp_dx_inlet_: torch.Tensor

# Build Parameters
def buildParams(device) -> Params:
    dtype       = torch.float32

    # Read Data
    df = None
    try: df = pd.read_csv(paths.data_dir / 'data.csv')
    except (FileNotFoundError, pd.errors.EmptyDataError): df = None
    with open(paths.data_dir / 'parameters.yaml', 'r') as file: params = yaml.safe_load(file)
    
    # Bifurcation Parameters
    R1          = torch.tensor(params["R1"], device=device, dtype=dtype)
    R2          = torch.tensor(params["R2"], device=device, dtype=dtype)
    R0          = torch.pow(R1**3 + R2**3, 1.0 / 3.0)
    # - 
    cos_α1      = (R0**4 + R1**4 - R2**4) / (2.0 * R0**2 * R1**2)
    cos_α2      = (R0**4 + R2**4 - R1**4) / (2.0 * R0**2 * R2**2)
    α1          = torch.acos(torch.clamp(cos_α1, -1.0, 1.0))
    α2          = torch.acos(torch.clamp(cos_α2, -1.0, 1.0))
    # - 
    LEN_PARENT  = torch.tensor(params["len_parent"], device=device, dtype=dtype)
    LEN_BRANCH  = torch.tensor(params["len_branch"], device=device, dtype=dtype)

    # From CSV
    u_data      = torch.tensor(df["u"].values, device=device, dtype=dtype)      # u data                      (m/s)
    u_max       = torch.tensor(u_data.max(), device=device, dtype=dtype)        # max u                       (m/s)
    
    # From YAML
    ϕ_max       = torch.tensor(params["phi_max"],  device=device, dtype=dtype)  # max ϕ                       (dimensionless)
    ϕ_bulk      = torch.tensor(params["phi_bulk"], device=device, dtype=dtype)  # bulk ϕ                      (dimensionless)
    p_max       = torch.tensor(params["p_max"],    device=device, dtype=dtype)  # max p                       ()
    ρ           = torch.tensor(params["rho"],      device=device, dtype=dtype)  # solvent density             (kg/m³)
    η           = torch.tensor(params["eta"],      device=device, dtype=dtype)  # dynamic viscosity           (Pa·s)
    Kn          = torch.tensor(params["Kn"],       device=device, dtype=dtype)  # fitting parameter           (dimensionless)
    λ2          = torch.tensor(params["lambda2"],  device=device, dtype=dtype)  # fitting parameter           (dimensionless)
    λ3          = torch.tensor(params["lambda3"],  device=device, dtype=dtype)  # fitting parameter           (dimensionless)
    α           = torch.tensor(params["alpha"],    device=device, dtype=dtype)  # α ∈ [2, 5]                  ()
    β           = torch.tensor(params["beta"],     device=device, dtype=dtype)
    a           = torch.tensor(params["a"],        device=device, dtype=dtype)  # particle radius             (m)  
    frv         = torch.tensor(params["frv"],      device=device, dtype=dtype)  # function of reduced volume  ()
    H0          = torch.tensor(params["H0"],       device=device, dtype=dtype)  # buffer                      ()
    dp_dx_inlet = torch.tensor(params["dp_dx_inlet"], device=device, dtype=dtype)  # inlet pressure gradient  ()
    
    # Derived quantities
    S           = R0                                                            # scaling metric              (m)
    η0          = η                                                             # dynamic viscosity alias     (Pa·s)
    ε           = a / ((S / 2.0) ** 2)                                          # non-local shear-rate coeff.
    dp_dx_inlet_= dp_dx_inlet * S**2 / (4 * η0 * u_max)                         # normalized dp_dx_inlet

    return Params(
        R0=R0,
        R1=R1,
        R2=R2,
        # -
        α1=α1,
        α2=α2,
        # -
        LEN_PARENT=LEN_PARENT,
        LEN_BRANCH=LEN_BRANCH,
        # -
        u_data=u_data, 
        u_max=u_max,
        ϕ_max=ϕ_max,
        ϕ_bulk=ϕ_bulk,
        p_max=p_max,
        # - 
        ρ=ρ,
        η=η,
        Kn=Kn,
        λ2=λ2,
        λ3=λ3,
        α=α,
        a=a,
        β=β,
        frv=frv,
        H0=H0,
        # - 
        S=S,
        η0=η0,
        ε=ε,
        dp_dx_inlet_=dp_dx_inlet_,
    )
