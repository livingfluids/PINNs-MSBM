import torch
import config
import geometry

# Initialize
Λ_migration_old      = 1.0
Λ_x_momentum_old     = 1.0
Λ_y_momentum_old     = 1.0
Λ_continuity_old     = 1.0
Λ_ϕ_continuity_old   = 1.0
Λ_no_slip_old        = 1.0
Λ_v_corner_old       = 1.0
Λ_u_data_old         = 1.0
Λ_v_data_old         = 1.0
Λ_ϕ_data_old         = 1.0
Λ_partition_ratio_old= 1.0
Λ_flow_flux_old      = 1.0
Λ_particle_flux_old  = 1.0
Λ_normal_particle_flux_wall_old = 1.0
Λ_ϕ_neumann_old      = 1.0
Λ_bulk_old           = 1.0
EPS                  = 1e-12

# Snapshot Of Adaptive Loss Weights (For Logging/Diagnostics)
def currentLossWeights():
    return {
        "migration": Λ_migration_old,
        "x_momentum": Λ_x_momentum_old,
        "y_momentum": Λ_y_momentum_old,
        "continuity": Λ_continuity_old,
        "no_slip": Λ_no_slip_old,
        "v_corner": Λ_v_corner_old,
        "u_data": Λ_u_data_old,
        "v_data": Λ_v_data_old,
        "ϕ_data": Λ_ϕ_data_old,
        "partition_ratio": Λ_partition_ratio_old,
        "flow_flux": Λ_flow_flux_old,
        "particle_flux": Λ_particle_flux_old,
        "normal_particle_flux_wall": Λ_normal_particle_flux_wall_old,
        "ϕ_neumann": Λ_ϕ_neumann_old,
        "bulk": Λ_bulk_old,
    }

def lineIntersection(first, second, dtype):
    p = first["x0"]
    r = first["x1"] - first["x0"]
    q = second["x0"]
    s = second["x1"] - second["x0"]

    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]

    denom = cross2(r, s)
    parallel_tol = 10.0 * torch.finfo(dtype).eps * torch.linalg.norm(r) * torch.linalg.norm(s)
    if torch.abs(denom) <= parallel_tol:
        raise ValueError("Could not locate the inner bifurcation V corner for DataLoss.")

    return p + (cross2(q - p, s) / denom) * r

def innerBifurcationVCorner(params, device, dtype):
    walls = geometry.bifurcationWalls(params=params, device=device, dtype=dtype)
    return lineIntersection(walls["top_minus"], walls["bottom_plus"], dtype=dtype)

# MSBM Equations Loss Terms | ∇⋅J = 0, (U⋅∇)U - ∇⋅Σ = 0, ∇⋅U = 0, Jn_Wall = 0 For Interior & Boundaries
def PDELoss(trials, params, array, device):
    xy_             = (array["full_array"].detach() / params.S).requires_grad_(True)  # [Nfull, 2]
    xy              = xy_ * params.S  # [Nfull, 2]
    x_              = xy_[:, 0:1]  # [Nfull, 1]
    y_              = xy_[:, 1:2]  # [Nfull, 1]
    # - 
    A               = params.a / params.S  # scalar
    zero            = torch.zeros_like(x_)  # [Nfull, 1]
    # - 
    u_, v_, p_, ϕ_  = trials.all_trials(xy_)  # [Nfull, 1] each
    # - 
    u_grad_         = grad(xy_, u_)  # [Nfull, 2]
    v_grad_         = grad(xy_, v_)  # [Nfull, 2]
    p_grad_         = grad(xy_, p_)  # [Nfull, 2]
    ϕ_grad_         = grad(xy_, ϕ_)
    du_dx_, du_dy_  = u_grad_[:, 0:1], u_grad_[:, 1:2]  # [Nfull, 1] each
    dv_dx_, dv_dy_  = v_grad_[:, 0:1], v_grad_[:, 1:2]  # [Nfull, 1] each
    dp_dx_, dp_dy_  = p_grad_[:, 0:1], p_grad_[:, 1:2]
    dϕ_dx_, dϕ_dy_  = ϕ_grad_[:, 0:1], ϕ_grad_[:, 1:2]
    
    # Masks
    segment_names = array["segment_names"]
    # - 
    inlet_id = segment_names.index("inlet")
    outlet_top_id = segment_names.index("outlet_top")
    outlet_bottom_id = segment_names.index("outlet_bottom")
    # - 
    boundary_mask   = array["is_boundary"]
    interior_mask   = array["is_interior"]

    # Velocity Field Gradient (∇U)
    U_grad_ = torch.stack([
        torch.cat([du_dx_, du_dy_, zero], dim=1),
        torch.cat([dv_dx_, dv_dy_, zero], dim=1),
        torch.cat([zero,   zero,   zero], dim=1)
    ], dim=1)  # [Nfull, 3, 3]

    # Strain Rate Tensor (E)
    E_ = 0.5 * (U_grad_ + U_grad_.transpose(1, 2))  # [Nfull, 3, 3]

    # Shear Rate Tensor (γ̇)
    γ̇_ = torch.sqrt(2 * torch.sum(E_ * E_, dim=(1, 2))).unsqueeze(1)  # [Nfull, 1]

    # Normal Stress Viscosity (ηₙ(ϕ))
    def ηN(ϕ): return params.Kn * (ϕ/params.ϕ_max)**2 * (1 - ϕ/params.ϕ_max)**(-2)  # torch.Size([y_.shape[0], 1]), a scalar for each y

    # Particle Phase Shear Viscosity (ηₚ(ϕ))
    def ηp(ϕ):
        ηs = (1 - ϕ/params.ϕ_max)**(-2)
        return ηs - 1  # torch.Size([y, 1]), a scalar for each y

    # Sedimentation Hinderence Function (f(ϕ))
    def f(ϕ): return (1 - ϕ/params.ϕ_max) * (1 - ϕ)**(params.α - 1)  # torch.Size([y, 1]), a scalar for each y

    # Lift Force (L)
    dist_to_wall, dir_to_wall = geometry.nearestSideWallGeometry(points=xy, params=params, array=array, device=device, dtype=params.u_max.dtype)
    γ̇           = γ̇_ * params.u_max / params.S  # dimensionalize for calculating it
    L           = ((3 * params.η0 * γ̇) / (4 * torch.pi * (dist_to_wall + params.H0)**params.β) * params.frv * (-dir_to_wall))  # [Nfull, 2]
    L__         = L * (params.S**2) / (params.η0 * params.u_max)  # nondimensionalize after calculating it
    L_          = torch.stack([L__[:, 0:1], L__[:, 1:2], torch.zeros_like(L__[:, 0:1])], dim=1)  # [Nfull, 3, 1]
    
    # v_corner = innerBifurcationVCorner(params=params, device=device, dtype=xy.dtype)
    # dist_to_corner = torch.sqrt(((xy - v_corner.unsqueeze(0))**2).sum(dim=1, keepdim=True))  # physical coords vs physical v_corner
    # corner_radius = params.R0 * 0.1
    # corner_mask = (dist_to_corner > corner_radius).float()  # [Nfull, 1]
    # L_ = L_ * corner_mask.unsqueeze(1)  # [Nfull, 1, 1] broadcasts with [Nfull, 3, 1]

    L_plot = (2 * A**2 / 9) * f(ϕ_).unsqueeze(1) * ϕ_.view(-1, 1, 1) * L_

    # Diagonal Tensor of the SBM (Q)
    Q = torch.tensor([
        [1.0, 0.0, 0.0], 
        [0.0, params.λ2, 0.0], 
        [0.0, 0.0, params.λ3]
        ], device=device, dtype=xy_.dtype).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Non-Local Shear Rate Tensor
    Umx_ = torch.maximum(torch.maximum(u_, v_), torch.zeros_like(u_))
    γ̇NL_ = params.ε * params.S * Umx_

    # Particle Normal Stress Diagonal Tensor (Σₙₙᵖ)
    # Σpnn_ = ηN(ϕ_).view(-1, 1, 1) * (γ̇_.unsqueeze(1) + γ̇NL_.view(-1, 1, 1)) * Q  # torch.Size([y, 3, 3]), a matrix for each y
    Σpnn_ = ηN(ϕ_).view(-1, 1, 1) * (γ̇_.unsqueeze(1)) * Q  # torch.Size([y, 3, 3]), a matrix for each y

    # Oriented Particle Stress Tensor (Σᵖ)
    Σp_ = -Σpnn_ + (2 * ηp(ϕ_).view(-1, 1, 1) * E_)  # torch.Size([y, 3, 3]), a matrix for each y

    # Divergence of the Oriented Particle Stress Tensor (∇⋅Σᵖ)
    dΣpxy_dy_   = grad(xy_, Σp_[:, 0, 1])[:, 1:2]  # ∂Σp_xy/∂y
    dΣpyx_dx_   = grad(xy_, Σp_[:, 1, 0])[:, 0:1]  # ∂Σp_yx/∂x
    dΣpxx_dx_   = grad(xy_, Σp_[:, 0, 0])[:, 0:1]
    dΣpyy_dy_   = grad(xy_, Σp_[:, 1, 1])[:, 1:2]
    Σp_div_     = torch.stack([
        dΣpxx_dx_ + dΣpxy_dy_ + zero,
        dΣpyx_dx_ + dΣpyy_dy_ + zero,   # now uses its own gradient
        torch.zeros_like(dΣpxx_dx_),
    ], dim=1)

    # Migration Flux (J)
    if config.DATA_DIR == 'MSBM_data': J_ = (2 * A**2 / 9) * f(ϕ_).unsqueeze(1) * (Σp_div_ + ϕ_.view(-1, 1, 1) * L_)
    elif config.DATA_DIR == 'SBM_data': J_ = (2 * A**2 / 9) * f(ϕ_).unsqueeze(1) * (Σp_div_)

    # Zero Normal Migration Flux BC (J⋅n = 0)
    segment_id_b = array["full_segment_id"][boundary_mask]
    inlet_mask_b = segment_id_b == inlet_id
    outlet_top_mask_b = segment_id_b == outlet_top_id
    outlet_bottom_mask_b = segment_id_b == outlet_bottom_id
    walls_mask_b = ~inlet_mask_b & ~outlet_top_mask_b & ~outlet_bottom_mask_b
    wall_normals = array["boundary_normals"][walls_mask_b]
    J_wall = J_[boundary_mask][:, 0:2, 0][walls_mask_b]
    Jn_wall = (J_wall * wall_normals).sum(dim=1, keepdim=True)

    # Identity Matrix (I)
    I = torch.eye(3, device=device, dtype=xy_.dtype).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid Phase Stress (Σᶠ)
    Σf_ = - p_.view(-1, 1, 1) * I + 2 * E_

    # Total Stress (Σ)
    Σ_ = Σp_ + Σf_

    # Navier-Stokes Stress Term (∇⋅Σ)
    dΣxy_       = grad(xy_, Σ_[:, 0, 1])
    dΣxy_dy_    = dΣxy_[:, 1:2]
    dΣyy_dy_    = grad(xy_, Σ_[:, 1, 1])[:, 1:2]
    # - 
    dΣyx_dx_    = dΣxy_[:, 0:1]
    dΣxx_dx_    = grad(xy_, Σ_[:, 0, 0])[:, 0:1]
    # - 
    Σx_div_     = dΣxy_dy_ + dΣxx_dx_
    Σy_div_     = dΣyx_dx_ + dΣyy_dy_
    # NOTE: do not need -∇p since p is now a scalar field inside Σf_

    # Navier-Stokes Momentum
    x_momentum = Σx_div_
    y_momentum = Σy_div_

    # Conservative Particle Transport Equation (∇⋅(ϕU + J) = 0)
    particle_flux_x_ = ϕ_ * u_ + J_[:, 0, 0:1]
    particle_flux_y_ = ϕ_ * v_ + J_[:, 1, 0:1]
    migration        = grad(xy_, particle_flux_x_)[:, 0:1] + grad(xy_, particle_flux_y_)[:, 1:2]

    # Continuity Equation (∇⋅U) 
    continuity  = du_dx_ + dv_dy_
    # -
    return migration[interior_mask], x_momentum[interior_mask], y_momentum[interior_mask], continuity[interior_mask], Jn_wall, L_plot

# No-Slip BCs | U = 0, For Boundaries
def noSlipBCLoss(trials, params, array, device):
    xy              = array["full_array"].detach().requires_grad_(True)  # [Nfull, 2]
    xy_             = xy / params.S  # [Nfull, 2]
    # - 
    u_, v_, p_, ϕ_  = trials.all_trials(xy_)  # [Nfull, 1] each
    # - 
    walls           = geometry.bifurcationWalls(params=params, device=device, dtype=xy_.dtype)
    wall_names      = [name for name in walls.keys() if ("plus" in name or "minus" in name)]
    segment_names   = array["segment_names"]
    full_segment_id = array["full_segment_id"]
    segment_losses  = []
    for wall_name in wall_names:
        if wall_name in segment_names:
            wall_mask = full_segment_id == segment_names.index(wall_name)
            if torch.any(wall_mask):
                wall_velocity = torch.cat([u_[wall_mask], v_[wall_mask]], dim=1)
                segment_losses.append(torch.mean(wall_velocity**2))
    # -
    if len(segment_losses) == 0:
        return (u_.sum() + v_.sum()).reshape(1) * 0.0
    # - 
    return torch.stack(segment_losses)

# Wall Volume-Fraction Neumann BC | ∂ϕ/∂n = 0, For Side Walls
def phiWallNeumannLoss(trials, params, array, device):
    xy_             = (array["full_array"].detach() / params.S).requires_grad_(True)  # [Nfull, 2]
    # -
    _, _, _, ϕ_     = trials.all_trials(xy_)
    ϕ_grad_         = grad(xy_, ϕ_)  # derivative wrt nondimensional coordinates; zero BC is unchanged by scaling
    # -
    segment_names   = array["segment_names"]
    full_segment_id = array["full_segment_id"]
    boundary_mask   = array["is_boundary"]
    segment_id_b    = full_segment_id[boundary_mask]
    # -
    inlet_id        = segment_names.index("inlet")
    outlet_top_id   = segment_names.index("outlet_top")
    outlet_bottom_id= segment_names.index("outlet_bottom")
    walls_mask_b    = (
        (segment_id_b != inlet_id)
        & (segment_id_b != outlet_top_id)
        & (segment_id_b != outlet_bottom_id)
    )
    # -
    wall_normals    = array["boundary_normals"][walls_mask_b]
    ϕ_grad_wall     = ϕ_grad_[boundary_mask][walls_mask_b]
    if ϕ_grad_wall.shape[0] == 0:
        return xy_.new_zeros((1, 1))
    # -
    return (ϕ_grad_wall[:, 0:2] * wall_normals).sum(dim=1, keepdim=True)

# V-Corner No-Slip BC | U = 0, At the Inner V-Corner Tip
def vCornerNoSlipBCLoss(trials, params, device):
    dtype           = params.u_max.dtype
    v_corner        = innerBifurcationVCorner(params=params, device=device, dtype=dtype)
    xy_corner_      = (v_corner / params.S).unsqueeze(0)
    # -
    u_, v_, _, _    = trials.all_trials(xy_corner_)
    # -
    return torch.cat([u_, v_], dim=1)

# Data Loss (Parent Branch Only)
def DataLoss(trials, params, array, device):
    My              = params.u_data.size(0)
    dtype           = params.u_data.dtype
    # - 
    v_corner        = innerBifurcationVCorner(params=params, device=device, dtype=dtype)
    profile_dist    = torch.as_tensor(config.DATA_LOSS_PROFILE_DISTANCE_FROM_V_CORNER, device=device, dtype=dtype)
    x_profile       = v_corner[0] - profile_dist
    y_data          = params.y_data.to(device=device, dtype=dtype).reshape(-1)
    # - 
    xy_data         = torch.stack([x_profile.expand_as(y_data), y_data], dim=1)  # [M, 2]
    xy_data_        = xy_data / params.S
    # - 
    u_, v_, p_, ϕ_  = trials.all_trials(xy_data_)  # [M, 1] each
    # - 
    u_data_         = (params.u_data / params.u_max).reshape(-1, 1)  # [M, 1]
    v_data_         = (params.v_data / params.u_max).reshape(-1, 1)  # [M, 1]
    ϕ_data_         = params.ϕ_data.reshape(-1, 1)  # [M, 1]
    # - 
    u_term          = u_data_ - u_  # [M, 1]
    v_term          = v_data_ - v_  # [M, 1]
    ϕ_term          = ϕ_data_ - ϕ_  # [M, 1]
    # - 
    phi_term        = ϕ_term
    # - 
    return u_term, v_term, phi_term

# Flow Partition Ratio
def flowPartitionRatio(trials, params, array, device):
    xy              = array["full_array"].detach().requires_grad_(True)  # [Nfull, 2]
    xy_             = xy / params.S  # [Nfull, 2]
    # - 
    boundary_mask   = array["is_boundary"]
    u_, v_, p_, ϕ_  = trials.all_trials(xy_)  # [Nfull, 1] each
    u_, v_, ϕ_      = u_[boundary_mask], v_[boundary_mask], ϕ_[boundary_mask]
    # - 
    normals = array["boundary_normals"]   # use this only if it is already boundary-only
    segment_id = array["full_segment_id"][boundary_mask]
    segment_names = array["segment_names"]
    # - 
    inlet_id = segment_names.index("inlet")
    outlet_top_id = segment_names.index("outlet_top")
    outlet_bottom_id = segment_names.index("outlet_bottom")
    # - 
    inlet_mask = (segment_id == inlet_id)
    outlet_top_mask = (segment_id == outlet_top_id)
    outlet_bot_mask = (segment_id == outlet_bottom_id)
    # - 
    U_norm = u_ * normals[:, 0:1] + v_ * normals[:, 1:2]
    # - 
    Nin = inlet_mask.sum()
    Ntop = outlet_top_mask.sum()
    Nbot = outlet_bot_mask.sum()
    
    # weight the fluxes according to the size of the respective outlet/inlet to better approximate the integral
    ds_in = 2.0 * params.R0 / (Nin + 1)
    ds_top = 2.0 * params.R1 / (Ntop + 1)
    ds_bot = 2.0 * params.R2 / (Nbot + 1)
    # - 
    flow_flux_in = -ds_in * torch.sum(U_norm[inlet_mask])
    flow_flux_top = ds_top * torch.sum(U_norm[outlet_top_mask])
    flow_flux_bottom = ds_bot * torch.sum(U_norm[outlet_bot_mask])
    # - 
    part_flux_in = -ds_in * (ϕ_[inlet_mask] * U_norm[inlet_mask]).sum()
    part_flux_top = ds_top * (ϕ_[outlet_top_mask] * U_norm[outlet_top_mask]).sum()
    part_flux_bottom = ds_bot * (ϕ_[outlet_bot_mask] * U_norm[outlet_bot_mask]).sum()
    # - 
    eps = torch.finfo(flow_flux_in.dtype).eps
    flow_flux_term = flow_flux_in - (flow_flux_top + flow_flux_bottom)
    part_flux_term = part_flux_in - (part_flux_top + part_flux_bottom)
    partition_term = torch.stack([
        flow_flux_top / (flow_flux_in + eps) - params.r,
        flow_flux_bottom / (flow_flux_in + eps) - (1 - params.r),
    ])
    # -  
    return flow_flux_term, part_flux_term, partition_term

# Particle Partition Ratio 
def particlePartitionRatio(trials, params, array, device):
    boundary_mask = array["is_boundary"]
    with torch.no_grad():
        xy = array["full_array"][boundary_mask]
        xy_ = xy / params.S
        # - 
        u_, v_, _, ϕ_ = trials.all_trials(xy_)
        normals = array["boundary_normals"]
        segment_id = array["full_segment_id"][boundary_mask]
        segment_names = array["segment_names"]
        # - 
        inlet_id = segment_names.index("inlet")
        outlet_top_id = segment_names.index("outlet_top")
        outlet_bottom_id = segment_names.index("outlet_bottom")
        # - 
        inlet_mask = (segment_id == inlet_id)
        outlet_top_mask = (segment_id == outlet_top_id)
        outlet_bot_mask = (segment_id == outlet_bottom_id)
        # - 
        un = u_ * normals[:, 0:1] + v_ * normals[:, 1:2]
        # - 
        Nin = inlet_mask.sum()
        Ntop = outlet_top_mask.sum()
        Nbot = outlet_bot_mask.sum()
        # - 
        ds_in = 2.0 * params.R0 / (Nin + 1)
        ds_top = 2.0 * params.R1 / (Ntop + 1)
        ds_bot = 2.0 * params.R2 / (Nbot + 1)
        # - 
        flux_in = -ds_in * (ϕ_[inlet_mask] * un[inlet_mask]).sum()
        flux_top = ds_top * (ϕ_[outlet_top_mask] * un[outlet_top_mask]).sum()
        flux_bottom = ds_bot * (ϕ_[outlet_bot_mask] * un[outlet_bot_mask]).sum()
        # - 
        eps = torch.finfo(flux_in.dtype).eps
        flux_top = flux_top / (flux_in + eps)
        flux_bottom = flux_bottom / (flux_in + eps)
        # - 
        return flux_top, flux_bottom

# Velocity-Weighted Bulk Conservation Loss Term
def BulkLoss(trials, params, array, device):
    xy_             = (array["full_array"].detach() / params.S).requires_grad_(True)  # [Nfull, 2]
    u_, v_, p_, ϕ_  = trials.all_trials(xy_)  # [Nfull, 1] each
    U_              = torch.sqrt(u_**2 + v_**2)
    bulk_term       = torch.sum(ϕ_ * U_) / torch.sum(U_) - params.ϕ_bulk
    # - 
    return bulk_term

# Total Loss (Max Absolute Gradient Normalization Scheme)
def totalLoss(trials, params, PINN, array, epoch, device):
    global Λ_migration_old, Λ_x_momentum_old, Λ_y_momentum_old, Λ_continuity_old, Λ_ϕ_continuity_old, Λ_no_slip_old, Λ_v_corner_old, Λ_u_data_old, Λ_v_data_old, Λ_ϕ_data_old, Λ_partition_ratio_old, Λ_flow_flux_old, Λ_particle_flux_old, Λ_normal_particle_flux_wall_old, Λ_ϕ_neumann_old, Λ_bulk_old
    # -
    migration, x_momentum, y_momentum, continuity, normal_particle_flux_wall, lift_force = PDELoss(trials, params, array, device)
    no_slip                                            = noSlipBCLoss(trials, params, array, device)
    ϕ_neumann_wall                                     = phiWallNeumannLoss(trials, params, array, device)
    v_corner_no_slip                                   = vCornerNoSlipBCLoss(trials, params, device)
    u_data, v_data, ϕ_data                             = DataLoss(trials, params, array, device)
    flow_flux, particle_flux, partition_ratio          = flowPartitionRatio(trials, params, array, device)
    bulk                                               = BulkLoss(trials, params, array, device)
    # -
    ℒ_migration_raw = torch.mean(migration**2)
    ℒ_x_momentum_raw= torch.mean(x_momentum**2)
    ℒ_y_momentum_raw= torch.mean(y_momentum**2)
    ℒ_continuity_raw= torch.mean(continuity**2)
    ℒ_normal_particle_flux_wall_raw = torch.mean(normal_particle_flux_wall**2)
    ℒ_ϕ_neumann_raw = torch.mean(ϕ_neumann_wall**2)
    ℒ_no_slip_raw   = torch.mean(no_slip)
    ℒ_v_corner_raw  = torch.mean(v_corner_no_slip**2)
    ℒ_u_data_raw    = torch.mean(u_data**2)
    ℒ_v_data_raw    = torch.mean(v_data**2)
    ℒ_ϕ_data_raw    = torch.mean(ϕ_data**2)
    ℒ_flow_flux_raw = torch.mean(flow_flux**2)
    ℒ_particle_flux_raw = torch.mean(particle_flux**2)
    ℒ_partition_ratio_raw = torch.mean(partition_ratio**2)
    ℒ_bulk_raw      = torch.mean(bulk**2)
    # - 
    if epoch % config.GRAD_NORM_EPOCH_INTERVAL == 0:
        max_gradient_migration  = maxGradMagnitude(ℒ_migration_raw, PINN)
        max_gradient_x_momentum = maxGradMagnitude(ℒ_x_momentum_raw, PINN)
        max_gradient_y_momentum = maxGradMagnitude(ℒ_y_momentum_raw, PINN)
        max_gradient_continuity = maxGradMagnitude(ℒ_continuity_raw, PINN)
        max_residual_gradient   = torch.max(torch.stack([max_gradient_migration, max_gradient_x_momentum, max_gradient_y_momentum, max_gradient_continuity]))
        # -
        gradient_migration      = meanAbsGrad(ℒ_migration_raw, PINN)
        gradient_x_momentum     = meanAbsGrad(ℒ_x_momentum_raw, PINN)
        gradient_y_momentum     = meanAbsGrad(ℒ_y_momentum_raw, PINN)
        gradient_continuity     = meanAbsGrad(ℒ_continuity_raw, PINN)
        gradient_normal_particle_flux_wall = meanAbsGrad(ℒ_normal_particle_flux_wall_raw, PINN)
        gradient_ϕ_neumann      = meanAbsGrad(ℒ_ϕ_neumann_raw, PINN)
        gradient_no_slip        = meanAbsGrad(ℒ_no_slip_raw, PINN)
        gradient_v_corner       = meanAbsGrad(ℒ_v_corner_raw, PINN)
        gradient_u_data         = meanAbsGrad(ℒ_u_data_raw, PINN)
        gradient_v_data         = meanAbsGrad(ℒ_v_data_raw, PINN)
        gradient_ϕ_data         = meanAbsGrad(ℒ_ϕ_data_raw, PINN)
        gradient_flow_flux      = meanAbsGrad(ℒ_flow_flux_raw, PINN)
        gradient_particle_flux  = meanAbsGrad(ℒ_particle_flux_raw, PINN)
        gradient_partition_ratio = meanAbsGrad(ℒ_partition_ratio_raw, PINN)
        gradient_bulk           = meanAbsGrad(ℒ_bulk_raw, PINN)
        # - 
        Λ_migration      = max_residual_gradient / (gradient_migration + EPS)
        Λ_x_momentum     = max_residual_gradient / (gradient_x_momentum + EPS)
        Λ_y_momentum     = max_residual_gradient / (gradient_y_momentum + EPS)
        Λ_continuity     = max_residual_gradient / (gradient_continuity + EPS)
        Λ_normal_particle_flux_wall = max_residual_gradient / (gradient_normal_particle_flux_wall + EPS)
        Λ_ϕ_neumann      = max_residual_gradient / (gradient_ϕ_neumann + EPS)
        Λ_no_slip        = max_residual_gradient / (gradient_no_slip + EPS)
        Λ_v_corner       = max_residual_gradient / (gradient_v_corner + EPS)
        Λ_u_data         = max_residual_gradient / (gradient_u_data + EPS)
        Λ_v_data         = max_residual_gradient / (gradient_v_data + EPS)
        Λ_ϕ_data         = max_residual_gradient / (gradient_ϕ_data + EPS)
        Λ_flow_flux      = max_residual_gradient / (gradient_flow_flux + EPS)
        Λ_particle_flux  = max_residual_gradient / (gradient_particle_flux + EPS)
        Λ_partition_ratio = max_residual_gradient / (gradient_partition_ratio + EPS)
        Λ_bulk           = max_residual_gradient / (gradient_bulk + EPS)
        # - 
        smoothing_factor = config.ξ
        Λ_migration_old      = (1 - smoothing_factor) * Λ_migration_old +       smoothing_factor * Λ_migration.item()
        Λ_x_momentum_old     = (1 - smoothing_factor) * Λ_x_momentum_old +      smoothing_factor * Λ_x_momentum.item()
        Λ_y_momentum_old     = (1 - smoothing_factor) * Λ_y_momentum_old +      smoothing_factor * Λ_y_momentum.item()
        Λ_continuity_old     = (1 - smoothing_factor) * Λ_continuity_old +      smoothing_factor * Λ_continuity.item()
        Λ_normal_particle_flux_wall_old = (1 - smoothing_factor) * Λ_normal_particle_flux_wall_old + smoothing_factor * Λ_normal_particle_flux_wall.item()
        Λ_ϕ_neumann_old      = (1 - smoothing_factor) * Λ_ϕ_neumann_old +       smoothing_factor * Λ_ϕ_neumann.item()
        Λ_no_slip_old        = (1 - smoothing_factor) * Λ_no_slip_old +         smoothing_factor * Λ_no_slip.item()
        Λ_v_corner_old       = (1 - smoothing_factor) * Λ_v_corner_old +        smoothing_factor * Λ_v_corner.item()
        Λ_u_data_old         = (1 - smoothing_factor) * Λ_u_data_old +          smoothing_factor * Λ_u_data.item()
        Λ_v_data_old         = (1 - smoothing_factor) * Λ_v_data_old +          smoothing_factor * Λ_v_data.item()
        Λ_ϕ_data_old         = (1 - smoothing_factor) * Λ_ϕ_data_old +          smoothing_factor * Λ_ϕ_data.item()
        Λ_flow_flux_old      = (1 - smoothing_factor) * Λ_flow_flux_old +       smoothing_factor * Λ_flow_flux.item()
        Λ_particle_flux_old  = (1 - smoothing_factor) * Λ_particle_flux_old +   smoothing_factor * Λ_particle_flux.item()
        Λ_partition_ratio_old = (1 - smoothing_factor) * Λ_partition_ratio_old + smoothing_factor * Λ_partition_ratio.item()
        Λ_bulk_old           = (1 - smoothing_factor) * Λ_bulk_old +            smoothing_factor * Λ_bulk.item()
        # - 
    ℒ_migration        = (ℒ_migration_raw * config.Λ_PDEs) * Λ_migration_old
    ℒ_x_momentum       = (ℒ_x_momentum_raw * config.Λ_PDEs) * Λ_x_momentum_old
    ℒ_y_momentum       = (ℒ_y_momentum_raw * config.Λ_PDEs) * Λ_y_momentum_old
    ℒ_continuity       = (ℒ_continuity_raw * config.Λ_PDEs) * Λ_continuity_old
    ℒ_normal_particle_flux_wall = (ℒ_normal_particle_flux_wall_raw * config.Λ_BCs) * Λ_normal_particle_flux_wall_old
    ℒ_ϕ_neumann        = (ℒ_ϕ_neumann_raw * config.Λ_BCs) * Λ_ϕ_neumann_old
    ℒ_no_slip          = (ℒ_no_slip_raw * config.Λ_BCs) * Λ_no_slip_old
    ℒ_v_corner         = (ℒ_v_corner_raw * config.Λ_BCs) * Λ_v_corner_old
    ℒ_u_data           = (ℒ_u_data_raw * config.Λ_data) * Λ_u_data_old
    ℒ_v_data           = (ℒ_v_data_raw * config.Λ_data) * Λ_v_data_old
    ℒ_ϕ_data           = (ℒ_ϕ_data_raw * config.Λ_data) * Λ_ϕ_data_old
    ℒ_flow_flux        = (ℒ_flow_flux_raw * config.Λ_BCs) * 100 # Λ_flow_flux_old
    ℒ_particle_flux    = (ℒ_particle_flux_raw * config.Λ_BCs) * 100 # Λ_particle_flux_old
    ℒ_partition_ratio  = (ℒ_partition_ratio_raw * config.Λ_BCs) * Λ_partition_ratio_old
    ℒ_bulk             = (ℒ_bulk_raw * config.Λ_PDEs) * Λ_bulk_old
    # - 
    total_loss = (
        ℒ_migration +
        ℒ_x_momentum +
        ℒ_y_momentum +
        ℒ_continuity +
        ℒ_no_slip +
        ℒ_v_corner + 
        ℒ_u_data +
        ℒ_v_data +
        ℒ_ϕ_data +
        ℒ_flow_flux +
        ℒ_particle_flux +
        ℒ_partition_ratio
        #ℒ_normal_particle_flux_wall
        # ℒ_ϕ_neumann
        )
    # - 
    return total_loss, lift_force

# Gradient Helper
def grad(y, f):
    y = y if y.ndim == 2 else y.unsqueeze(1)
    df_dy = torch.autograd.grad(f, y, torch.ones_like(f), create_graph=True, allow_unused=True)[0]
    if df_dy is None: df_dy = torch.zeros_like(y)
    # - 
    return df_dy

# Max Gradient Helper Function
def maxGradMagnitude(loss, PINN):
    param_list = list(PINN.parameters())
    grads = torch.autograd.grad(loss, param_list, retain_graph=True, create_graph=False, allow_unused=True)
    max_abs = 0.0
    for g in grads:
        if g is not None:
            gmax = torch.max(torch.abs(g))
            if torch.isfinite(gmax): max_abs = max(max_abs, gmax.item())
    return torch.tensor(max_abs, device=loss.device) + EPS

# Mean Absolute Gradient Helper Function
def meanAbsGrad(loss, PINN):
    param_list = list(PINN.parameters())
    grads = torch.autograd.grad(loss, param_list, retain_graph=True, create_graph=False, allow_unused=True)
    total_abs = 0.0
    count = 0
    for g in grads:
        if g is not None:
            g_abs = torch.abs(g)
            finite_mask = torch.isfinite(g_abs)
            if finite_mask.any():
                total_abs += torch.sum(g_abs[finite_mask])
                count += int(finite_mask.sum().item())
    if count == 0: return torch.tensor(EPS, device=loss.device)
    return (total_abs / count).clamp_min(EPS)
