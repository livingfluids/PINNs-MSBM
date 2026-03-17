import torch
import config
import geometry

"""
ϕ BulkLoss enforced at inlet and outlets only
No J flux through walls 
everywhere ∇⋅Σ = 0, ∇⋅J = 0, ∇⋅U = 0
no slip at walls for velocity
pressure gradient imposed inside the parent tube only

lift force:
- dist field wrt nearest wall
- lift force dir is the normalized gradient of the dist field
"""

# Gradient Helper
def grad(y, f):
    y = y if y.ndim == 2 else y.unsqueeze(1)
    df_dy = torch.autograd.grad(f, y, torch.ones_like(f), create_graph=True, allow_unused=True)[0]
    if df_dy is None:
        df_dy = torch.zeros_like(y)
    return df_dy

# MSBM Equations Loss Terms | ∇⋅J = 0, (U⋅∇)U - ∇⋅Σ = 0, ∇⋅U = 0
def PDELoss(trials, params, array, device):
    xy              = array["full_array"][array["is_interior"]].detach().requires_grad_(True)  # torch.Size([xy, 2])
    xy_             = xy / params.S  # torch.Size([xy, 2])
    u_              = trials.u_trial(xy_)  # torch.Size([xy, 1])
    v_              = trials.v_trial(xy_)  # torch.Size([xy, 1])
    p_              = trials.p_trial(xy_)  # torch.Size([xy, 1])
    ϕ_              = trials.ϕ_trial(xy_)  # torch.Size([xy, 1])
    x_              = xy_[:,0:1]  # torch.Size([xy, 1])
    y_              = xy_[:,1:2]  # torch.Size([xy, 1])
    u_grad_         = grad(xy_, u_)  # torch.Size([xy, 2])
    v_grad_         = grad(xy_, v_)  # torch.Size([xy, 2])
    p_grad_         = grad(xy_, p_)  # torch.Size([xy, 2])
    du_dx_, du_dy_  = u_grad_[:,0:1], u_grad_[:,1:2]  # torch.Size([xy, 1]) each
    dv_dx_, dv_dy_  = v_grad_[:,0:1], v_grad_[:,1:2]  # torch.Size([xy, 1]) each
    dp_dx_, dp_dy_  = p_grad_[:,0:1], p_grad_[:,1:2]
    A               = params.a / params.S  # scalar 
    zero            = torch.zeros_like(x_, device=device)  # torch.Size([y, 1])

    # Normal stress viscosity (ηₙ(ϕ))
    def ηN(ϕ): return params.Kn * (ϕ/params.ϕ_max)**2 * (1 - ϕ/params.ϕ_max)**(-2)  # torch.Size([y_.shape[0], 1]), a scalar for each y

    # Shear viscosity of the particle phase (ηₚ(ϕ))
    def ηp(ϕ):
        ηs = (1 - ϕ/params.ϕ_max)**(-2)
        return ηs - 1  # torch.Size([y, 1]), a scalar for each y

    # Sedimentation hinderence function for mobility of particle phase (f(ϕ))
    def f(ϕ): return (1 - ϕ/params.ϕ_max) * (1 - ϕ)**(params.α - 1)  # torch.Size([y, 1]), a scalar for each y

    # Gradient of the velocity field (∇U)
    U_grad_ = torch.stack([
        torch.cat([du_dx_, du_dy_, zero], dim=1),
        torch.cat([dv_dx_, dv_dy_, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Strain rate tensor (E)
    E_ = 0.5 * (U_grad_ + U_grad_.transpose(1, 2))  # torch.Size([y, 3, 3]), a matrix for each y

    # Shear rate tensor (γ̇)
    γ̇_ = torch.sqrt(2 * torch.sum(E_ * E_, dim=(1, 2))).unsqueeze(1)  # torch.Size([y, 1])

    # Lift Force (L)
    γ̇ = γ̇_ * params.u_max / params.S  # dimensionalize for calculating it
    L = 3 * params.η0 * γ̇ / (4 * torch.pi * (array["distance_to_nearest_boundary"][array["is_interior"]] + params.H0)**params.β) * params.frv * array["direction_to_nearest_boundary"][array["is_interior"]]
    L__ = L * (params.S**2) / (params.η0 * params.u_max)  # nondimensionalize after calculating it 
    L_ = torch.stack([L__[:, 0:1], L__[:, 1:2], torch.zeros_like(L__[:, 0:1])], dim=1)  # [N,3,1]
    
    # Diagonal tensor of the SBM (Q)
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, params.λ2, 0.0], [0.0, 0.0, params.λ3]], device=device, dtype=xy_.dtype).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Non-local shear rate tensor
    γ̇NL_ = params.ε * params.S / 2

    # Particle normal stress diagonal tensor (Σₙₙᵖ)
    Σpnn_ = ηN(ϕ_).view(-1, 1, 1) * (γ̇_.unsqueeze(1) + γ̇NL_) * Q  # torch.Size([y, 3, 3]), a matrix for each y

    # Oriented particle stress tensor (Σᵖ)
    Σp_ = -Σpnn_ + (2 * ηp(ϕ_).view(-1, 1, 1) * E_)  # torch.Size([y, 3, 3]), a matrix for each y

    # Divergence of oriented particle stress tensor (∇⋅Σᵖ)
    dΣpxx_dx_ = grad(xy_, Σp_[:, 0, 0])[:, 0:1]
    dΣpxy_ = grad(xy_, Σp_[:, 0, 1])
    dΣpxy_dx_ = dΣpxy_[:, 0:1]
    dΣpxy_dy_ = dΣpxy_[:, 1:2]
    dΣpyy_dy_ = grad(xy_, Σp_[:, 1, 1])[:, 1:2]
    Σp_div_ = torch.stack([
        dΣpxx_dx_ + dΣpxy_dy_ + zero,
        dΣpxy_dx_ + dΣpyy_dy_ + zero,
        torch.zeros_like(dΣpxx_dx_),
    ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y

    # Migration flux (J)
    J_ = - (2 * A**2 / 9) * f(ϕ_).unsqueeze(1) * (Σp_div_ + ϕ_.view(-1, 1, 1) * L_)  # torch.Size([y, 3, 1])

    # Soft enforce zero migration flux at walls
    # Omitted for now 

    # Divergence of Migration Flux (∇⋅J)
    dJz_dz_ = zero
    dJx_dx_ = grad(xy_, J_[:, 0, 0])[:, 0:1]
    dJy_dy_ = grad(xy_, J_[:, 1, 0])[:, 1:2]
    J_div_ = dJx_dx_ + dJy_dy_ + dJz_dz_  # torch.Size([y, 1])

    # Identity Matrix (I)
    I = torch.eye(3, device=device, dtype=xy_.dtype).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid phase Stress (Σᶠ)
    Σf_ = - p_.view(-1, 1, 1) * I + 2 * E_

    # Total Stress (Σ)
    Σ_ = Σp_ + Σf_

    # Navier-Stokes Stress Term (∇⋅Σ)
    dΣxy_ = grad(xy_, Σ_[:, 0, 1])
    dΣxy_dy_ = dΣxy_[:, 1:2]
    dΣyy_dy_ = grad(xy_, Σ_[:, 1, 1])[:, 1:2]
    # - 
    dΣyx_dx_ = dΣxy_[:, 0:1]
    dΣxx_dx_ = grad(xy_, Σ_[:, 0, 0])[:, 0:1]
    # - 
    Σx_div_ = dΣxy_dy_ + dΣxx_dx_ # - dp_dx_
    Σy_div_ = dΣyx_dx_ + dΣyy_dy_ # - dp_dy_
    # NOTE: do not need -∇p since p is now a scalar field inside Σf_

    # Navier-Stokes Convective Term ((U⋅∇)U)
    Re_ = 1 #params.ρ * params.u_max * params.S / params.η0
    U_u_grad_ = Re_ * (u_ * du_dx_ + v_ * du_dy_)
    U_v_grad_ = Re_ * (u_ * dv_dx_ + v_ * dv_dy_)

    # Navier-Stokes Momentum Equation ((U⋅∇)U = ∇⋅Σ)
    x_momentum = U_u_grad_ - Σx_div_
    y_momentum = U_v_grad_ - Σy_div_

    # Continuity Equation (∇⋅U) 
    xy              = array["full_array"].detach().requires_grad_(True)  # torch.Size([xy, 2])
    xy_             = xy / params.S  # torch.Size([xy, 2])
    u_              = trials.u_trial(xy_)  # torch.Size([xy, 1])
    v_              = trials.v_trial(xy_)  # torch.Size([xy, 1])
    u_grad_         = grad(xy_, u_)  # torch.Size([xy, 2])
    v_grad_         = grad(xy_, v_)  # torch.Size([xy, 2])
    du_dx_, _  = u_grad_[:,0:1], u_grad_[:,1:2]  # torch.Size([xy, 1]) each
    _, dv_dy_  = v_grad_[:,0:1], v_grad_[:,1:2]  # torch.Size([xy, 1]) each
    # - 
    continuity = du_dx_ + dv_dy_

    # Migration Flux Term
    migration = J_div_

    return migration, x_momentum, y_momentum, continuity

# No-Slip BCs
def noSlipBCLoss(trials, params, array, device):
    xy              = array["full_array"].detach().requires_grad_(True)  # torch.Size([xy, 2])
    xy_             = xy / params.S  # torch.Size([xy, 2])
    u_              = trials.u_trial(xy_)  # torch.Size([xy, 1])
    v_              = trials.v_trial(xy_)  # torch.Size([xy, 1])
    walls           = geometry.bifurcationWalls(params=params, device=device, dtype=xy_.dtype)
    wall_names      = [name for name in walls.keys() if ("plus" in name or "minus" in name)]
    segment_names   = array["segment_names"]
    full_segment_id = array["full_segment_id"]
    wall_masks      = {
        name: (full_segment_id == segment_names.index(name))
        if name in segment_names
        else torch.zeros_like(full_segment_id, dtype=torch.bool, device=full_segment_id.device)
        for name in wall_names
    }
    # - 
    wall_terms      = []
    for wall_name in wall_names:
        wall_mask   = wall_masks[wall_name]
        wall_term   = torch.abs(u_[wall_mask]) + torch.abs(v_[wall_mask])
        wall_terms.append(torch.mean(wall_term))
    wall_loss_term  = torch.mean(torch.stack(wall_terms))
    # - 
    return wall_loss_term

# Inlet Pressure-Drive Flow BC
def pressureBCLoss(trials, params, array, device):
    xy              = array["full_array"].detach().requires_grad_(True)  # torch.Size([xy, 2])
    xy_             = xy / params.S  # torch.Size([xy, 2])
    segment_names   = array["segment_names"]
    full_segment_id = array["full_segment_id"]
    inlet_mask      = (full_segment_id == segment_names.index("inlet")) if "inlet" in segment_names else torch.zeros_like(full_segment_id, dtype=torch.bool, device=full_segment_id.device)
    p_              = trials.p_trial(xy_)  # torch.Size([xy, 1])
    p_grad_         = grad(xy_, p_)  # torch.Size([xy, 2])
    dp_dx_, dp_dy_  = p_grad_[:,0:1], p_grad_[:,1:2]  # torch.Size([xy, 1]) each
    u_              = trials.u_trial(xy_)  # torch.Size([xy, 1])
    v_              = trials.v_trial(xy_)  # torch.Size([xy, 1])
    # - 
    inlet_term = abs(params.u_max - u_[inlet_mask]) + abs(v_[inlet_mask])
    # - 
    return inlet_term

# u-Data Loss (Parent Branch Only, e.g., at x = LEN_PARENT outlet cross-section)
def uDataLoss(trials, params, array, device):
    M = params.u_data.size(0)
    y_data_ = torch.linspace(-params.R0 / params.S, params.R0 / params.S, steps=M, device=device, dtype=torch.float32)  # [M]
    x_data_ = torch.linspace(0.0 * params.LEN_PARENT / params.S, 0.8 * params.LEN_PARENT / params.S, steps=M, device=device, dtype=torch.float32)  # [M]
    
    # Create full grid: [M, M] for x and y, then flatten to [M*M, 2] for trial input
    x_grid, y_grid = torch.meshgrid(x_data_, y_data_, indexing='ij')  # [M, M] each
    xy_data_ = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)  # [M*M, 2]
    # - 
    u_ = trials.u_trial(xy_data_)  # [M, 1]
    v_ = trials.v_trial(xy_data_)  # [M, 1]
    # - 
    u_data_ = (params.u_data / params.u_max).unsqueeze(1)  # [M, 1]
    u_data_repeated = u_data_.repeat(M, 1).flatten().unsqueeze(1)  # [M*M, 1]
    # - 
    u_term = torch.abs(u_data_repeated - u_)  # [M, 1]
    v_term = torch.abs(v_)  # [M, 1]
    # - 
    velocity_term = torch.mean(u_term + v_term)
    # - 
    return velocity_term

# Velocity-Weighted Bulk Conservation Loss Term (Parent Branch Only)
def ϕBulkLoss(trials, params, array, device):
    xy              = array["full_array"].detach().requires_grad_(True)[array["is_interior"]]  # torch.Size([xy, 2])
    xy_             = (xy / params.S) #[(xy[:,0:1] < params.LEN_PARENT).squeeze()]  # torch.Size([xy, 2])
    u_              = trials.u_trial(xy_)  # torch.Size([xy, 1])
    v_              = trials.v_trial(xy_)  # torch.Size([xy, 1])
    U_mag_          = torch.sqrt(u_**2 + v_**2)
    ϕ_              = trials.ϕ_trial(xy_)  # torch.Size([xy, 1])
    bulk_term       = torch.sum(ϕ_ * U_mag_) / torch.sum(U_mag_) - params.ϕ_bulk
    # - 
    return bulk_term

# Global Mass Conservation Loss (AV_in = AV_out)
def contLoss(trials, params, array, device):
    boundary_mask = array["is_boundary"]

    xy = array["full_array"].detach().requires_grad_(True)[boundary_mask]
    xy_ = xy / params.S

    u_ = trials.u_trial(xy_)
    v_ = trials.v_trial(xy_)

    normals = array["boundary_normals"]   # use this only if it is already boundary-only
    segment_id = array["full_segment_id"][boundary_mask]
    segment_names = array["segment_names"]

    inlet_id = segment_names.index("inlet")
    outlet_top_id = segment_names.index("outlet_top")
    outlet_bottom_id = segment_names.index("outlet_bottom")

    inlet_mask = (segment_id == inlet_id)
    outlet_top_mask = (segment_id == outlet_top_id)
    outlet_bot_mask = (segment_id == outlet_bottom_id)

    un = u_ * normals[:, 0:1] + v_ * normals[:, 1:2]

    flux_in = -torch.sum(un[inlet_mask])
    flux_top = torch.sum(un[outlet_top_mask])
    flux_bottom = torch.sum(un[outlet_bot_mask])

    return flux_in - (flux_top + flux_bottom)