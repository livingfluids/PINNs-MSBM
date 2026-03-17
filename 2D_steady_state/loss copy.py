import torch
import config
import geometry

"""
ϕ BulkLoss enforced at inlet and outlets only
No J flux through walls 
everywhere ∇⋅Σ = 0, ∇⋅J = 0, ∇⋅U = 0
no slip at walls for velocity
pressure gradient imposed inside the parent tube only
"""


def grad(y, f):
    y = y if y.ndim == 2 else y.unsqueeze(1)
    df_dy = torch.autograd.grad(f, y, torch.ones_like(f), create_graph=True, allow_unused=True)[0]
    if df_dy is None:
        df_dy = torch.zeros_like(y)
    return df_dy

def wallBoundaryMask(array, exclude_caps=True):
    seg_names = array["segment_names"]
    seg_id_b = array["boundary_segment_id"]
    # -
    if exclude_caps:
        excluded = {"inlet", "outlet_top", "outlet_bottom"}
    else:
        excluded = set()
    wall_ids = [i for i, name in enumerate(seg_names) if name not in excluded]
    # -
    if len(wall_ids) == 0:
        return torch.zeros_like(seg_id_b, dtype=torch.bool)
    wall_id_tensor = torch.tensor(wall_ids, device=seg_id_b.device, dtype=seg_id_b.dtype)
    return (seg_id_b[:, None] == wall_id_tensor[None, :]).any(dim=1)

def segmentBoundaryMask(array, segment_name):
    seg_names = array["segment_names"]
    seg_id_b = array["boundary_segment_id"]
    if segment_name not in seg_names:
        return torch.zeros_like(seg_id_b, dtype=torch.bool)
    seg_id = seg_names.index(segment_name)
    return seg_id_b == seg_id

def distToWall(xy, x0, x1):
    eps = 1e-12
    # -
    wall_dir_vect = x1 - x0
    wall_dir_vect_sqrd = torch.sum(wall_dir_vect ** 2) + eps
    # -
    proj_param = (((xy - x0) @ wall_dir_vect) / wall_dir_vect_sqrd)
    proj_param = torch.clamp(proj_param, 0.0, 1.0).unsqueeze(1)
    # -
    closest_xy = x0.unsqueeze(0) + proj_param * wall_dir_vect.unsqueeze(0)
    xy_to_wall_vect = xy - closest_xy
    xy_to_wall_dist = torch.sqrt(torch.sum(xy_to_wall_vect * xy_to_wall_vect, dim=1, keepdim=True) + eps)
    xy_to_wall_vect_unit = xy_to_wall_vect / (xy_to_wall_dist + eps)
    # - 
    return xy_to_wall_dist, xy_to_wall_vect_unit


def _wall_names(walls):
    return [name for name in walls.keys() if ("plus" in name or "minus" in name)]

# Cache Helper
def cacheHelper(trials, params, array, device):
    xy              = array["full_array"].detach().requires_grad_(True)
    xy_             = 2 * xy / params.H
    u_, v_, p_, ϕ_  = trials.uvpϕ_trial(xy_)
    # - 
    walls           = geometry.bifurcationWalls(params=params, device=device, dtype=xy_.dtype)
    wall_names      = _wall_names(walls)
    wall_masks      = {name: geometry.exteriorBoundaryMask(array, segment_name=name) for name in wall_names}
    return {
        "xy": xy,
        "xy_": xy_,
        "u_": u_,
        "v_": v_,
        "p_": p_,
        "ϕ_": ϕ_,
        "walls": walls,
        "wall_names": wall_names,
        "wall_masks": wall_masks,
    }

# Lift force helper; returns dimensional lift vector (x,y).
def liftFromWalls(xy, γ̇, params, device, dtype, walls=None, wall_names=None, length_scale=None):
    if walls is None:
        walls = geometry.bifurcationWalls(params=params, device=device, dtype=dtype)
    if wall_names is None:
        wall_names = _wall_names(walls)
    if length_scale is None:
        length_scale = params.H
    L_ref = torch.clamp(torch.as_tensor(length_scale, device=xy.device, dtype=xy.dtype), min=1e-12)
    # - 
    xy_to_walls_dist_list = []
    xy_to_walls_vect_unit_list = []
    for wall_name in wall_names:
        wall = walls[wall_name]
        xy_to_wall_dist, xy_to_wall_vect_unit = distToWall(xy=xy, x0=wall["x0"], x1=wall["x1"])
        xy_to_walls_dist_list.append(xy_to_wall_dist)
        xy_to_walls_vect_unit_list.append(xy_to_wall_vect_unit)

    D = torch.cat(xy_to_walls_dist_list, dim=1)        # (N, Nseg), dimensional distance
    E = torch.stack(xy_to_walls_vect_unit_list, dim=2) # (N, 2, Nseg)
    # -
    D_nd = D / L_ref
    W = torch.softmax(-D_nd, dim=1)  # largest weight for closest wall
    eff_dist = torch.sum(W * D, dim=1, keepdim=True)
    eff_dir = torch.sum(E * W.unsqueeze(1), dim=2)
    # -
    eff_dist_nd = eff_dist / L_ref
    H0_nd = torch.as_tensor(params.H0, device=xy.device, dtype=xy.dtype) / L_ref
    lift_mag_dim = 3 * params.η0 * γ̇ * params.frv / (4 * torch.pi * L_ref * (eff_dist_nd + H0_nd)**params.β)
    L_xy = lift_mag_dim * eff_dir
    # - 
    return L_xy

def JAtWalls(array, J_, walls, wall_names, wall_masks=None):
    # - 
    wall_terms = []
    for wall_name in wall_names:
        if wall_masks is None: wall_mask = geometry.segmentBoundaryMask(array, segment_name=wall_name)
        else: wall_mask = wall_masks[wall_name]
        n = walls[wall_name]["normal"]  # shape (2,)
        Jx = J_[wall_mask, 0, 0]
        Jy = J_[wall_mask, 1, 0]
        Jn = Jx * n[0] + Jy * n[1]
        wall_terms.append(torch.mean(Jn**2))  # or torch.mean(torch.abs(Jn))
    wall_loss_term = torch.mean(torch.stack(wall_terms))
    # -
    return wall_loss_term

# MSBM Equations Loss Terms | ∇⋅J = ∇⋅Σ = 0
def PDELoss(trials, params, array, device, cache=None):
    if cache is None: cache = cacheHelper(trials=trials, params=params, array=array, device=device)
    xy          = cache["xy"]
    xy_         = cache["xy_"]
    u_          = cache["u_"]
    v_          = cache["v_"]
    p_          = cache["p_"]
    ϕ_          = cache["ϕ_"]
    walls       = cache["walls"]
    wall_names  = cache["wall_names"]
    wall_masks  = cache["wall_masks"]
    # -
    x_          = xy_[:,0:1]
    y_          = xy_[:,1:2]
    # - 
    u_grad_     = grad(xy_, u_)  # sizes
    v_grad_     = grad(xy_, v_)
    # p_grad_     = grad(xy_, p_)
    # - 
    du_dx_, du_dy_  = u_grad_[:,0:1], u_grad_[:,1:2]  # sizes
    dv_dx_, dv_dy_  = v_grad_[:,0:1], v_grad_[:,1:2]
    # dp_dx_, dp_dy_  = p_grad_[:,0:1], p_grad_[:,1:2]
    # - 
    A           = 2 * params.a / params.H_phys
    # -
    zero        = torch.zeros_like(x_, device=device)  # torch.Size([y, 1])

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

    # Lift force (L): wall-distance based contribution from bifurcation walls.
    # Use the same reference length as coordinate normalization to avoid scale mismatch.
    L_ref = torch.clamp(params.H, min=1e-12)
    γ̇ = γ̇_ * 2 * params.u_max / L_ref
    L_xy = liftFromWalls(
        xy=xy,
        γ̇=γ̇,
        params=params,
        device=device,
        dtype=xy_.dtype,
        walls=walls,
        wall_names=wall_names,
        length_scale=L_ref,
    )
    L_xy_ = L_xy * (L_ref ** 2) / (2 * params.η0 * params.u_max + 1e-12)
    L = torch.stack([
        torch.cat([L_xy_[:,0:1]], dim=1),
        torch.cat([L_xy_[:,1:2]], dim=1),
        torch.cat([zero], dim=1)
    ], dim=1)
    
    # Diagonal tensor of the SBM (Q)
    Q = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, params.λ2, 0.0], [0.0, 0.0, params.λ3]],
        device=device,
        dtype=xy_.dtype,
    ).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Non-local shear rate tensor
    γ̇NL_ = params.ε * params.H_phys / 2

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
        torch.cat([dΣpxx_dx_ + dΣpxy_dy_ + zero], dim=1),
        torch.cat([dΣpxy_dx_ + dΣpyy_dy_ + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y

    # Migration flux (J)
    J_ = - (2 * A**2 / 9) * f(ϕ_).unsqueeze(1) * (Σp_div_ + ϕ_.view(-1, 1, 1) * L)  # torch.Size([y, 3, 1])

    # Soft enforce zero migration flux at walls
    J_wall_ = JAtWalls(array=array, J_=J_, walls=walls, wall_names=wall_names, wall_masks=wall_masks)

    # Divergence of migration flux (∇⋅J)
    dJz_dz_ = zero
    dJx_dx_ = grad(xy_, J_[:, 0, 0])[:, 0:1]
    dJy_dy_ = grad(xy_, J_[:, 1, 0])[:, 1:2]
    J_div_ = dJx_dx_ + dJy_dy_ + dJz_dz_  # torch.Size([y, 1])

    # Identity matrix (I)
    I = torch.eye(3, device=device, dtype=xy_.dtype).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid phase stress (Σᶠ)
    Σf_ = - p_.view(-1, 1, 1) * I + 2 * E_

    # Total stress (Σ)
    Σ_ = Σp_ + Σf_

    # Suspension momentum balance (∇⋅Σ)
    dΣxy_ = grad(xy_, Σ_[:, 0, 1])
    dΣxy_dy_ = dΣxy_[:, 1:2]
    dΣyy_dy_ = grad(xy_, Σ_[:, 1, 1])[:, 1:2]
    # - 
    dΣyx_dx_ = dΣxy_[:, 0:1]
    dΣxx_dx_ = grad(xy_, Σ_[:, 0, 0])[:, 0:1]
    # - 
    Σx_div_ = dΣxy_dy_ + dΣxx_dx_
    Σy_div_ = dΣyx_dx_ + dΣyy_dy_
    # NOTE: do not need -∇p since p is now a scalar field inside Σf_

    Re_ = params.ρ * params.u_max * (params.H / 2.0) / params.η0
    x_momentum = Re_ * (u_ * du_dx_ + v_ * du_dy_) - Σx_div_
    y_momentum = Re_ * (u_ * dv_dx_ + v_ * dv_dy_) - Σy_div_

    # Continuity Equation Incompressible
    continuity = du_dx_ + dv_dy_

    return J_div_, J_wall_, x_momentum, y_momentum, continuity

# Inlet Pressure Gradient 
def inletPressureGradientLoss(trials, params, array, device, shared=None):
    if shared is None: shared = cacheHelper(trials=trials, params=params, array=array, device=device)
    xy = shared["xy"]
    xy_ = shared["xy_"]
    p_ = shared["p_"]
    p_grad_ = grad(xy_, p_)
    dp_dx_, dp_dy_ = p_grad_[:,0:1], p_grad_[:,1:2]
    # - 
    target_dp_dx_inlet_ = params.dp_dx_inlet_
    parent_mask = geometry.insideRect(
        coll_array=xy,
        position=(0.0, 0.0),
        angle=0.0,
        width=params.LEN_PARENT,
        height=2.0 * params.R0,
    )
    if not torch.any(parent_mask):
        return torch.zeros((0, 1), device=device, dtype=xy_.dtype)
    dp_dx_parent_ = dp_dx_[parent_mask]
    inlet_p_grad_term = torch.abs(target_dp_dx_inlet_ - dp_dx_parent_) + torch.abs(dp_dy_[parent_mask])
    # - 
    return inlet_p_grad_term

# No Slip At Walls 
def noSlipWalls(trials, params, array, device, shared=None):
    if shared is None:
        shared = cacheHelper(trials=trials, params=params, array=array, device=device)
    u_ = shared["u_"]
    v_ = shared["v_"]
    wall_names = shared["wall_names"]
    wall_masks = shared["wall_masks"]
    # - 
    wall_terms = []
    for wall_name in wall_names:
        wall_mask = wall_masks[wall_name]
        wall_term = torch.abs(u_[wall_mask]) + torch.abs(v_[wall_mask])
        wall_terms.append(torch.mean(wall_term))
    wall_loss_term = torch.mean(torch.stack(wall_terms))
    # - 
    return wall_loss_term

# Velocity-Weighted Bulk Conservation Loss Term
def ϕBulkLoss(trials, params, array, device, shared=None):
    if shared is None:
        shared = cacheHelper(trials=trials, params=params, array=array, device=device)
    xy_ = shared["xy_"]
    u_ = shared["u_"]
    v_ = shared["v_"]
    ϕ_ = shared["ϕ_"]
    eps = 1e-12
    # -

    cap_names = ["inlet", "outlet_top", "outlet_bottom"]
    segment_names = array["segment_names"]
    segment_id_full = array["segment_id"]
    boundary_normals = array["boundary_normals"]
    n_coll = array["coll_array"].shape[0]
    
    ub = u_[n_coll:]
    vb = v_[n_coll:]
    ϕb = ϕ_[n_coll:]

    un = ub * boundary_normals[:, 0:1] + vb * boundary_normals[:, 1:2]
    w = torch.abs(un)
    # -
    ϕ_bulk_terms = []
    for name in cap_names:
        if name not in segment_names:
            continue
        seg_idx = segment_names.index(name)
        mask_full = segment_id_full == seg_idx
        mask_b = mask_full[n_coll:]
        if not torch.any(mask_b):
            continue
        denom = torch.sum(w[mask_b])
        if torch.abs(denom) < eps:
            continue
        ϕ_cap = torch.sum(ϕb[mask_b] * w[mask_b]) / (denom + eps)
        ϕ_bulk_terms.append(ϕ_cap - params.ϕ_bulk)
    # -
    if len(ϕ_bulk_terms) == 0:
        return torch.zeros((), device=device, dtype=xy_.dtype)
    return torch.mean(torch.stack(ϕ_bulk_terms))


def computeLossTerms(trials, params, array, device):
    shared = cacheHelper(trials=trials, params=params, array=array, device=device)
    J_div_, J_wall_, Σx_div_, Σy_div_, continuity = PDELoss(
        trials=trials,
        params=params,
        array=array,
        device=device,
        cache=shared,
    )
    noslip_loss = noSlipWalls(
        trials=trials,
        params=params,
        array=array,
        device=device,
        shared=shared,
    )
    inlet_grad_loss = inletPressureGradientLoss(
        trials=trials,
        params=params,
        array=array,
        device=device,
        shared=shared,
    )
    ϕ_bulk_loss = ϕBulkLoss(
        trials=trials,
        params=params,
        array=array,
        device=device,
        shared=shared,
    )
    return J_div_, J_wall_, Σx_div_, Σy_div_, continuity, noslip_loss, inlet_grad_loss, ϕ_bulk_loss
