import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Callable, Optional
import yaml
import geometry
import config
import pandas as pd
import paths

def _segmentLineIntersection(first, second, dtype):
    p = first["x0"]
    r = first["x1"] - first["x0"]
    q = second["x0"]
    s = second["x1"] - second["x0"]
    # - 
    def cross2(a, b): return a[0] * b[1] - a[1] * b[0]
    denom = cross2(r, s)
    # - 
    return p + (cross2(q - p, s) / denom) * r

def _innerVCornerOffset(R0, R1, R2, α1, α2, LEN_BRANCH, device, dtype):
    tmp = type("GeometryParams", (), {})()
    tmp.R0 = R0
    tmp.R1 = R1
    tmp.R2 = R2
    tmp.α1 = α1
    tmp.α2 = α2
    tmp.LEN_PARENT = torch.tensor(0.0, device=device, dtype=dtype)
    tmp.LEN_BRANCH = LEN_BRANCH
    # - 
    walls = geometry.bifurcationWalls(params=tmp, device=device, dtype=dtype)
    v_corner = _segmentLineIntersection(walls["top_minus"], walls["bottom_plus"], dtype=dtype)
    # - 
    return v_corner[0]

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
    profile_dist: torch.Tensor
    # - 
    r: torch.Tensor
    # -
    u_data: torch.Tensor
    v_data: torch.Tensor
    y_data: torch.Tensor
    u_max: torch.Tensor
    ϕ_data: torch.Tensor
    ϕ_data_dghr: torch.Tensor
    ϕ_data_prnt: torch.Tensor
    U_data_dghr: torch.Tensor
    U_data_prnt: torch.Tensor
    L_mag_dghr: torch.Tensor
    L_mag_prnt: torch.Tensor
    L_mag_mask: torch.Tensor
    L_mag_mask_prnt: torch.Tensor
    dghr_valid_mask: torch.Tensor
    prnt_valid_mask: torch.Tensor
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
    dghr_start: torch.Tensor
    dghr_skew: bool 
    dghr_profile_source: str
    dghr_profile_reference: str
    openfoam_x_origin: torch.Tensor
    dghr_xy_local: torch.Tensor
    # - 
    S: torch.Tensor
    η0: torch.Tensor
    ε: torch.Tensor
    dp_dx_inlet_: torch.Tensor
    dghr_eta: Optional[torch.Tensor] = None
    dghr_arc: Optional[torch.Tensor] = None
    prnt_eta: Optional[torch.Tensor] = None
    prnt_arc: Optional[torch.Tensor] = None

# Build Parameters
def buildParams(device) -> Params:
    dtype       = torch.float32

    # Read Data
    df_inlet    = pd.read_csv(paths.data_dir / 'data.csv')
    df_daughter = pd.read_csv(paths.data_dir / 'data_daughter.csv')
    df_parent   = pd.read_csv(paths.data_dir / 'data_parent.csv')
    with open(paths.data_dir / 'parameters.yaml', 'r') as file: params = yaml.safe_load(file)
    
    # Bifurcation Parameters
    R1          = torch.tensor(params["R1"], device=device, dtype=dtype)  # top daughter branch radius (m)
    R2          = torch.tensor(params["R2"], device=device, dtype=dtype)  # bottom daughter branch radius (m)
    R0          = torch.pow(R1**3 + R2**3, 1.0 / 3.0)  # parent branch radius (m)
    # - 
    cos_α1      = (R0**4 + R1**4 - R2**4) / (2.0 * R0**2 * R1**2)  # cos(α1)
    cos_α2      = (R0**4 + R2**4 - R1**4) / (2.0 * R0**2 * R2**2)  # cos(α2)
    α1          = torch.acos(torch.clamp(cos_α1, -1.0, 1.0))  # top daughter branch angle w.r.t centerline (radians)
    α2          = torch.acos(torch.clamp(cos_α2, -1.0, 1.0))  # bottom daughter branch angle w.r.t centerline (radians)
    # - 
    LEN_PARENT  = torch.tensor(params["len_parent"], device=device, dtype=dtype)  # parent branch length (m)
    LEN_BRANCH  = torch.tensor(params["len_branch"], device=device, dtype=dtype)  # daughter branch length (m)
    profile_dist= torch.tensor(config.DATA_LOSS_PROFILE_DISTANCE_FROM_V_CORNER, device=device, dtype=dtype)
    v_offset    = _innerVCornerOffset(R0, R1, R2, α1, α2, LEN_BRANCH, device, dtype)
    LEN_PARENT  = profile_dist - v_offset
    # - 
    r           = torch.tensor(params["r"], device=device, dtype=dtype)  # flow partition ratio (dimensionless)

    # From CSV
    u_data      = torch.tensor(df_inlet["u"].values, device=device, dtype=dtype) # u data (m/s)
    v_data      = torch.tensor(df_inlet["v"].values, device=device, dtype=dtype) # v data (m/s)
    y_data_col  = "Points_1" if "Points_1" in df_inlet.columns else "y"
    y_data      = torch.tensor(df_inlet[y_data_col].values, device=device, dtype=dtype) # y-coordinate data (m)
    u_max       = u_data.max().detach().clone().to(device=device, dtype=dtype)  # max u (m/s)
    ϕ_data      = torch.tensor(df_inlet["phi"].values, device=device, dtype=dtype)  # ϕ data (dimensionless)
    ϕ_data_dghr = torch.tensor(df_daughter["phi"].values, device=device, dtype=dtype)  # ϕ profile data (dimensionless)
    ϕ_data_prnt = torch.tensor(df_parent["phi"].values, device=device, dtype=dtype)  # ϕ profile data (dimensionless)
    U_data_dghr = torch.tensor(np.sqrt(df_daughter["u"].values**2 + df_daughter["v"].values**2), device=device, dtype=dtype)
    U_data_prnt = torch.tensor(np.sqrt(df_parent["u"].values**2 + df_parent["v"].values**2), device=device, dtype=dtype)
    # - 
    L_mag_dghr  = torch.tensor(df_daughter["LiftF_Magnitude"].values, device=device, dtype=dtype)
    L_mag_prnt  = torch.tensor(df_parent["LiftF_Magnitude"].values, device=device, dtype=dtype)
    
    if "vtkValidPointMask" in df_daughter.columns: dghr_valid_mask = torch.tensor(df_daughter["vtkValidPointMask"].values, device=device, dtype=torch.bool)
    else: dghr_valid_mask = torch.ones(len(df_daughter), device=device, dtype=torch.bool)
    if "vtkValidPointMask" in df_parent.columns: prnt_valid_mask = torch.tensor(df_parent["vtkValidPointMask"].values, device=device, dtype=torch.bool)
    else: prnt_valid_mask = torch.ones(len(df_parent), device=device, dtype=torch.bool)
    
    L_mag_mask  = dghr_valid_mask
    L_mag_mask_prnt = prnt_valid_mask
    
    # From YAML
    ϕ_max       = torch.tensor(params["phi_max"],  device=device, dtype=dtype)  # max ϕ (dimensionless)
    ϕ_bulk      = torch.tensor(params["phi_bulk"], device=device, dtype=dtype)  # bulk ϕ (dimensionless)
    p_max       = torch.tensor(params["p_max"],    device=device, dtype=dtype)  # max p ()
    ρ           = torch.tensor(params["rho"],      device=device, dtype=dtype)  # solvent density (kg/m³)
    η           = torch.tensor(params["eta"],      device=device, dtype=dtype)  # dynamic viscosity (Pa·s)
    Kn          = torch.tensor(params["Kn"],       device=device, dtype=dtype)  # fitting parameter (dimensionless)
    λ2          = torch.tensor(params["lambda2"],  device=device, dtype=dtype)  # fitting parameter (dimensionless)
    λ3          = torch.tensor(params["lambda3"],  device=device, dtype=dtype)  # fitting parameter (dimensionless)
    α           = torch.tensor(params["alpha"],    device=device, dtype=dtype)  # α ∈ [2, 5] ()
    β           = torch.tensor(params["beta"],     device=device, dtype=dtype)  # MSBM exponent ()
    a           = torch.tensor(params["a"],        device=device, dtype=dtype)  # particle radius (m)  
    frv         = torch.tensor(params["frv"],      device=device, dtype=dtype)  # function of reduced volume ()
    H0          = torch.tensor(params["H0"],       device=device, dtype=dtype)  # buffer ()
    dp_dx_inlet = torch.tensor(params["dp_dx_inlet"], device=device, dtype=dtype)  # inlet pressure gradient ()
    # - 
    dghr_start  = torch.tensor(params["daughter_profile_start"], device=device, dtype=dtype)
    dghr_arc    = torch.tensor(df_daughter["arc_length"].values, device=device, dtype=dtype)
    dghr_skew   = bool(params.get("daughter_profile_skewed", False))
    dghr_profile_source = str(params.get("daughter_profile_source", "geometry")).strip().lower()
    dghr_profile_reference = str(params.get("daughter_profile_start_reference", "branch_origin")).strip().lower()
    
    openfoam_x_origin_value = params.get("openfoam_x_origin")
    if openfoam_x_origin_value is None: openfoam_x_origin_value = float(df_inlet["Points_0"].mean()) if "Points_0" in df_inlet.columns else 0.0
    openfoam_x_origin = torch.tensor(openfoam_x_origin_value, device=device, dtype=dtype)
    
    dghr_xy_local = torch.tensor(df_daughter[["Points_0", "Points_1"]].values, device=device, dtype=dtype)
    dghr_xy_local = dghr_xy_local.clone()
    dghr_xy_local[:, 0] = dghr_xy_local[:, 0] - openfoam_x_origin
    dghr_eta    = dghr_arc if dghr_skew else dghr_arc - R1
    prnt_eta    = torch.tensor(df_parent["arc_length"].values, device=device, dtype=dtype) - R0
    prnt_arc    = torch.tensor(df_parent["arc_length"].values, device=device, dtype=dtype)

    # Derived quantities
    S           = R0  # scaling metric (m)
    η0          = η  # dynamic viscosity alias (Pa·s)
    ε           = torch.tensor(params["epsilon"], device=device, dtype=dtype)
    dp_dx_inlet_= dp_dx_inlet * S**2 / (4 * η0 * u_max)  # normalized dp_dx_inlet

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
        profile_dist=profile_dist,
        # - 
        r=r,
        # - 
        u_data=u_data,
        v_data=v_data,
        y_data=y_data,
        u_max=u_max,
        ϕ_data=ϕ_data,
        ϕ_data_dghr=ϕ_data_dghr,
        ϕ_data_prnt=ϕ_data_prnt,
        U_data_dghr=U_data_dghr,
        U_data_prnt=U_data_prnt,
        L_mag_dghr=L_mag_dghr,
        L_mag_prnt=L_mag_prnt,
        L_mag_mask=L_mag_mask,
        L_mag_mask_prnt=L_mag_mask_prnt,
        dghr_valid_mask=dghr_valid_mask,
        prnt_valid_mask=prnt_valid_mask,
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
        dghr_start=dghr_start,
        dghr_eta=dghr_eta,
        dghr_arc=dghr_arc,
        prnt_eta=prnt_eta,
        prnt_arc=prnt_arc,
        dghr_skew=dghr_skew,
        dghr_profile_source=dghr_profile_source,
        dghr_profile_reference=dghr_profile_reference,
        openfoam_x_origin=openfoam_x_origin,
        dghr_xy_local=dghr_xy_local,
        # - 
        S=S,
        η0=η0,
        ε=ε,
        dp_dx_inlet_=dp_dx_inlet_,
    )
