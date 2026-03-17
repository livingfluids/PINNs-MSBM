# Calculate Gradient Helper Function
import torch
import config
import geometry

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

def distToSegment(xy, x0, x1):
    eps = 1e-12
    # -
    v = x1 - x0
    vv = torch.sum(v * v) + eps
    # -
    t = ((xy - x0) @ v) / vv
    t = torch.clamp(t, 0.0, 1.0).unsqueeze(1)
    # -
    closest = x0.unsqueeze(0) + t * v.unsqueeze(0)
    dvec = xy - closest
    dist = torch.sqrt(torch.sum(dvec * dvec, dim=1, keepdim=True) + eps)
    edir = dvec / (dist + eps)  # from wall toward point
    return dist, edir

def liftFromWalls(xy, γ̇, params, device, dtype, include_caps=False):
    # Smoothly combines lift contributions from all wall segments.
    segments = geometry.bifurcationWalls(params=params, device=device, dtype=dtype)
    # -
    if include_caps:
        names = list(segments.keys())
    else:
        names = [name for name in segments.keys() if ("plus" in name or "minus" in name)]

    if len(names) == 0:
        return torch.zeros((xy.shape[0], 2), device=device, dtype=dtype)

    d_list = []
    e_list = []
    for name in names:
        seg = segments[name]
        d_i, e_i = distToSegment(xy=xy, x0=seg["x0"], x1=seg["x1"])
        d_list.append(d_i)
        e_list.append(e_i)

    D = torch.cat(d_list, dim=1)        # (N, Nseg)
    E = torch.stack(e_list, dim=2)      # (N, 2, Nseg)
    # -
    k_soft = torch.as_tensor(getattr(params, "k_lift_soft", 20.0), device=device, dtype=dtype)
    W = torch.softmax(-k_soft * D, dim=1)
    d_eff = torch.sum(W * D, dim=1, keepdim=True)
    e_eff = torch.sum(E * W.unsqueeze(1), dim=2)
    # -
    lift_mag_dim = 3 * params.η0 * γ̇ * params.frv / (4 * torch.pi * (d_eff + params.H0)**params.β)
    scale_L = (params.H ** 2) / (2 * params.η0 * params.u_max + 1e-12)
    L_xy = scale_L * lift_mag_dim * e_eff
    return L_xy

# MSBM Equations Loss Terms | ∇⋅J = ∇⋅Σ = 0
def PDELoss(trials, params, array, device):
    xy_ = array["full_array"].clone().detach().requires_grad_(True)  # need to normalize this
    x_ = xy_[:,0:1]
    y_ = xy_[:,1:2]
    # - 
    u_ = trials.u_trial(xy_)
    v_ = trials.v_trial(xy_)
    p_ = trials.p_trial(xy_)
    ϕ_ = trials.ϕ_trial(xy_)
    # - 
    u_grad_ = grad(xy_, u_)  # sizes
    v_grad_ = grad(xy_, v_)
    p_grad_ = grad(xy_, p_)
    # - 
    du_dx_, du_dy_ = u_grad_[:,0:1], u_grad_[:,1:2]  # sizes
    dv_dx_, dv_dy_ = v_grad_[:,0:1], v_grad_[:,1:2]
    dp_dx_, dp_dy_ = p_grad_[:,0:1], p_grad_[:,1:2]
    # - 
    A = 2 * params.a / params.H
    # -
    zero = torch.zeros_like(x_, device=device)  # torch.Size([y, 1])

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
    γ̇ = γ̇_ * 2 * params.u_max / params.H
    L_xy = liftFromWalls(xy=xy_, γ̇=γ̇, params=params, device=device, dtype=xy_.dtype, include_caps=False)
    L = torch.stack([
        torch.cat([L_xy[:,0:1]], dim=1),
        torch.cat([L_xy[:,1:2]], dim=1),
        torch.cat([zero], dim=1)
    ], dim=1)
    
    # Diagonal tensor of the SBM (Q)
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, params.λ2, 0.0], [0.0, 0.0, params.λ3]], device=device).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Non-local shear rate tensor
    γ̇NL_ = params.ε * params.H / 2

    # Particle normal stress diagonal tensor (Σₙₙᵖ)
    Σpnn_ = ηN(ϕ_).view(-1, 1, 1) * (γ̇_.unsqueeze(1) + γ̇NL_) * Q  # torch.Size([y, 3, 3]), a matrix for each y

    # Oriented particle stress tensor (Σᵖ)
    Σp_ = -Σpnn_ + (2 * ηp(ϕ_).view(-1, 1, 1) * E_)  # torch.Size([y, 3, 3]), a matrix for each y

    # Divergence of oriented particle stress tensor (∇⋅Σᵖ)
    dΣpxx_dx_ = grad(xy_, Σp_[:, 0, 0])[:, 0:1]
    dΣpxy_dx_ = grad(xy_, Σp_[:, 0, 1])[:, 0:1]
    dΣpxy_dy_ = grad(xy_, Σp_[:, 0, 1])[:, 1:2]
    dΣpyy_dy_ = grad(xy_, Σp_[:, 1, 1])[:, 1:2]
    Σp_div_ = torch.stack([
        torch.cat([dΣpxx_dx_ + dΣpxy_dy_ + zero], dim=1),
        torch.cat([dΣpxy_dx_ + dΣpyy_dy_ + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y

    # Migration flux (J)
    J_ = - (2 * A**2 / 9) * f(ϕ_).unsqueeze(1) * (Σp_div_ + ϕ_.view(-1, 1, 1) * L)  # torch.Size([y, 3, 1])

    # Soft enforce zero normal migration flux on wall boundaries only
    n_coll = array["coll_array"].shape[0]
    Jb_x = J_[n_coll:, 0, 0:1]
    Jb_y = J_[n_coll:, 1, 0:1]
    normals_b = array["boundary_normals"]
    # -
    Jn_b = Jb_x * normals_b[:, 0:1] + Jb_y * normals_b[:, 1:2]
    # -
    seg_names = array["segment_names"]
    seg_id_b = array["boundary_segment_id"]
    excluded = {"inlet", "outlet_top", "outlet_bottom"}
    wall_ids = [i for i, name in enumerate(seg_names) if name not in excluded]
    # -
    if len(wall_ids) > 0:
        wall_id_tensor = torch.tensor(wall_ids, device=seg_id_b.device, dtype=seg_id_b.dtype)
        wall_mask = (seg_id_b[:, None] == wall_id_tensor[None, :]).any(dim=1)
        J_wall_ = Jn_b[wall_mask]
    else: J_wall_ = torch.empty((0, 1), device=device, dtype=xy_.dtype)

    # Inlet pressure-gradient BC: dp/dx at parent inlet.
    p_grad_b = p_grad_[n_coll:, :]
    inlet_mask = segmentBoundaryMask(array=array, segment_name="inlet")
    if inlet_mask.any():
        dpdx_inlet_ = p_grad_b[inlet_mask, 0:1] - params.dp_dx_inlet
    else:
        dpdx_inlet_ = torch.empty((0, 1), device=device, dtype=xy_.dtype)

    # Divergence of migration flux (∇⋅J)
    dJz_dz_ = zero
    dJx_dx_ = grad(xy_, J_[:, 0, 0])[:, 0:1]
    dJy_dy_ = grad(xy_, J_[:, 1, 0])[:, 1:2]
    J_div_ = dJx_dx_ + dJy_dy_ + dJz_dz_  # torch.Size([y, 1])

    # Identity matrix (I)
    I = torch.eye(3, device=device).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid phase stress (Σᶠ)
    Σf_ = - p_.view(-1, 1, 1) * I + 2 * E_

    # Total stress (Σ)
    Σ_ = Σp_ + Σf_

    # Suspension momentum balance (∇⋅Σ)
    dΣxy_dy_ = grad(xy_, Σ_[:, 0, 1])[:, 1:2]
    dΣyy_dy_ = grad(xy_, Σ_[:, 1, 1])[:, 1:2]
    # - 
    dΣyx_dx_ = grad(xy_, Σ_[:, 0, 1])[:, 0:1]
    dΣxx_dx_ = grad(xy_, Σ_[:, 0, 0])[:, 0:1]
    # - 
    Σx_div_ = dΣxy_dy_ + dΣxx_dx_ - dp_dx_
    Σy_div_ = dΣyx_dx_ + dΣyy_dy_ - dp_dy_
    # NOTE: might not need -∇p since p is now a scalar FIELD inside Σf_

    # Continuity Equation Incompressible
    continuity = du_dx_ + dv_dy_

    return J_div_, J_wall_, dpdx_inlet_, Σx_div_, Σy_div_, continuity

# Velocity-weighted bulk conservation in 2D interior.
def ϕBulkLoss(trials, params, array):
    xy_c = array["coll_array"]
    # -
    u_c = trials.u_trial(xy_c)
    v_c = trials.v_trial(xy_c)
    ϕ_c = trials.ϕ_trial(xy_c)
    # -
    # Keep same idea as 1D velocity-weighted bulk, but use in-plane speed in 2D.
    w = torch.sqrt(u_c.detach()**2 + v_c.detach()**2 + 1e-12)
    ϕ_bulk_term = torch.sum(ϕ_c * w) / (torch.sum(w) + 1e-12) - params.ϕ_bulk
    return ϕ_bulk_term

# 2D wall BC for phi (typically drives phi -> 0 on walls / CFL regions).
def ϕBCLoss(trials, params, array, wall_only=True):
    case = getattr(config, "CASE", None)
    if case is not None and case != "learn beta":
        raise ValueError("ϕBCLoss is only defined for CASE='learn beta'")
    # -
    xy_b = array["boundary_array"]
    if xy_b.shape[0] == 0:
        return torch.empty((0, 1), device=xy_b.device, dtype=xy_b.dtype)
    # -
    ϕ_b = trials.ϕ_trial(xy_b)
    if not wall_only:
        return ϕ_b
    wall_mask = wallBoundaryMask(array=array, exclude_caps=True)
    return ϕ_b[wall_mask]

# 2D velocity BC on walls: slip/no-penetration => u·n = 0.
def uWallBCLoss(trials, array, wall_only=True):
    xy_b = array["boundary_array"]
    if xy_b.shape[0] == 0:
        return torch.empty((0, 1), device=xy_b.device, dtype=xy_b.dtype)
    # -
    u_b = trials.u_trial(xy_b)
    v_b = trials.v_trial(xy_b)
    n_b = array["boundary_normals"]
    un_b = u_b * n_b[:, 0:1] + v_b * n_b[:, 1:2]
    if not wall_only:
        return un_b
    wall_mask = wallBoundaryMask(array=array, exclude_caps=True)
    return un_b[wall_mask]

# Safe Ratio Helper Function
EPS = getattr(config, "EPS", 1e-12)
ADAPTIVE_WEIGHT_MIN = getattr(config, "ADAPTIVE_WEIGHT_MIN", 1e-3)
ADAPTIVE_WEIGHT_MAX = getattr(config, "ADAPTIVE_WEIGHT_MAX", 1e3)

def safeRatio(num, den):
    return num / (den + EPS)

def clampAdaptiveWeight(weight):
    return torch.clamp(weight, min=ADAPTIVE_WEIGHT_MIN, max=ADAPTIVE_WEIGHT_MAX)

# Max Gradient Helper Function
def maxGradMagnitude(loss, PINN, params):
    if not loss.requires_grad:
        return torch.tensor(EPS, device=loss.device)
    param_list = list(PINN.parameters()) # + [params.β]
    grads = torch.autograd.grad(loss, param_list, retain_graph=True, create_graph=False, allow_unused=True)
    max_abs = 0.0
    for g in grads:
        if g is not None:
            gmax = torch.max(torch.abs(g))
            if torch.isfinite(gmax): max_abs = max(max_abs, gmax.item())
    return torch.tensor(max_abs, device=loss.device) + EPS

# Mean Absolute Gradient Helper Function
def meanAbsGrad(loss, PINN, params):
    if not loss.requires_grad:
        return torch.tensor(EPS, device=loss.device)
    param_list = list(PINN.parameters()) # + [params.β]
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

def _mean_sq(t, device):
    if t.numel() == 0:
        return torch.zeros((), device=device)
    return torch.mean(t**2)

def lossGradientNormalizationMax(trials, params, PINN, array, epoch, device):
    global Λ_J_old, Λ_Jwall_old, Λ_uWall_old, Λ_dpIn_old, Λ_Σx_old, Λ_Σy_old, Λ_cont_old, Λ_mass_old, Λ_ϕBC_old
    include_ϕ_BC = (getattr(config, "CASE", None) == "learn beta")
    # -
    J_div_, J_wall_, dpdx_inlet_, Σx_div_, Σy_div_, continuity_ = PDELoss(trials=trials, params=params, array=array, device=device)
    u_wall_term = uWallBCLoss(trials=trials, array=array, wall_only=True)
    ϕ_bulk_term = ϕBulkLoss(trials=trials, params=params, array=array)
    ϕ_BC_term = ϕBCLoss(trials=trials, params=params, array=array, wall_only=True) if include_ϕ_BC else torch.empty((0, 1), device=device)
    # -
    ℒ_J_raw = _mean_sq(J_div_, device=device)
    ℒ_Jwall_raw = _mean_sq(J_wall_, device=device)
    ℒ_uWall_raw = _mean_sq(u_wall_term, device=device)
    ℒ_dpIn_raw = _mean_sq(dpdx_inlet_, device=device)
    ℒ_Σx_raw = _mean_sq(Σx_div_, device=device)
    ℒ_Σy_raw = _mean_sq(Σy_div_, device=device)
    ℒ_cont_raw = _mean_sq(continuity_, device=device)
    ℒ_mass_raw = ϕ_bulk_term**2
    ℒ_ϕBC_raw = _mean_sq(ϕ_BC_term, device=device)
    # -
    if "Λ_J_old" not in globals():
        Λ_J_old = torch.tensor(1.0, device=device)
        Λ_Jwall_old = torch.tensor(1.0, device=device)
        Λ_uWall_old = torch.tensor(1.0, device=device)
        Λ_dpIn_old = torch.tensor(1.0, device=device)
        Λ_Σx_old = torch.tensor(1.0, device=device)
        Λ_Σy_old = torch.tensor(1.0, device=device)
        Λ_cont_old = torch.tensor(1.0, device=device)
        Λ_mass_old = torch.tensor(1.0, device=device)
        Λ_ϕBC_old = torch.tensor(1.0, device=device)
    # -
    interval = int(getattr(config, "GRAD_NORM_EPOCH_INTERVAL", 1))
    ξ = float(getattr(config, "ξ", 0.9))
    # -
    if epoch % interval == 0:
        g_J = maxGradMagnitude(ℒ_J_raw, PINN, params)
        g_Σx = maxGradMagnitude(ℒ_Σx_raw, PINN, params)
        g_Σy = maxGradMagnitude(ℒ_Σy_raw, PINN, params)
        g_cont = maxGradMagnitude(ℒ_cont_raw, PINN, params)
        g_ref = torch.max(torch.stack([g_J, g_Σx, g_Σy, g_cont]))
        # -
        g_Jwall = meanAbsGrad(ℒ_Jwall_raw, PINN, params)
        g_uWall = meanAbsGrad(ℒ_uWall_raw, PINN, params)
        g_dpIn = meanAbsGrad(ℒ_dpIn_raw, PINN, params)
        g_mass = meanAbsGrad(ℒ_mass_raw, PINN, params)
        if include_ϕ_BC: g_ϕBC = meanAbsGrad(ℒ_ϕBC_raw, PINN, params)
        else: g_ϕBC = torch.tensor(EPS, device=device)
        # -
        Λ_J = clampAdaptiveWeight(safeRatio(g_ref, g_J))
        Λ_Jwall = clampAdaptiveWeight(safeRatio(g_ref, g_Jwall))
        Λ_uWall = clampAdaptiveWeight(safeRatio(g_ref, g_uWall))
        Λ_dpIn = clampAdaptiveWeight(safeRatio(g_ref, g_dpIn))
        Λ_Σx = clampAdaptiveWeight(safeRatio(g_ref, g_Σx))
        Λ_Σy = clampAdaptiveWeight(safeRatio(g_ref, g_Σy))
        Λ_cont = clampAdaptiveWeight(safeRatio(g_ref, g_cont))
        Λ_mass = clampAdaptiveWeight(safeRatio(g_ref, g_mass))
        Λ_ϕBC = clampAdaptiveWeight(safeRatio(g_ref, g_ϕBC))
        # -
        Λ_J_old = ξ * Λ_J_old + (1 - ξ) * Λ_J
        Λ_Jwall_old = ξ * Λ_Jwall_old + (1 - ξ) * Λ_Jwall
        Λ_uWall_old = ξ * Λ_uWall_old + (1 - ξ) * Λ_uWall
        Λ_dpIn_old = ξ * Λ_dpIn_old + (1 - ξ) * Λ_dpIn
        Λ_Σx_old = ξ * Λ_Σx_old + (1 - ξ) * Λ_Σx
        Λ_Σy_old = ξ * Λ_Σy_old + (1 - ξ) * Λ_Σy
        Λ_cont_old = ξ * Λ_cont_old + (1 - ξ) * Λ_cont
        Λ_mass_old = ξ * Λ_mass_old + (1 - ξ) * Λ_mass
        Λ_ϕBC_old = ξ * Λ_ϕBC_old + (1 - ξ) * Λ_ϕBC
    # -
    Λ_PDEs = float(getattr(config, "Λ_PDEs", 1.0))
    Λ_BCs = float(getattr(config, "Λ_BCs", 1.0))
    # -
    ℒ_J = (ℒ_J_raw * Λ_PDEs) * Λ_J_old
    ℒ_Jwall = (ℒ_Jwall_raw * Λ_BCs) * Λ_Jwall_old
    ℒ_uWall = (ℒ_uWall_raw * Λ_BCs) * Λ_uWall_old
    ℒ_dpIn = (ℒ_dpIn_raw * Λ_BCs) * Λ_dpIn_old
    ℒ_Σx = (ℒ_Σx_raw * Λ_PDEs) * Λ_Σx_old
    ℒ_Σy = (ℒ_Σy_raw * Λ_PDEs) * Λ_Σy_old
    ℒ_cont = (ℒ_cont_raw * Λ_PDEs) * Λ_cont_old
    ℒ_mass = (ℒ_mass_raw * Λ_BCs) * Λ_mass_old
    ℒ_ϕBC = (ℒ_ϕBC_raw * Λ_BCs) * Λ_ϕBC_old if include_ϕ_BC else torch.zeros((), device=device)
    # -
    if include_ϕ_BC:
        ℒ = ℒ_J + ℒ_Jwall + ℒ_uWall + ℒ_dpIn + ℒ_Σx + ℒ_Σy + ℒ_cont + ℒ_mass + ℒ_ϕBC
        ℒ_no_weights = ℒ_J_raw + ℒ_Jwall_raw + ℒ_uWall_raw + ℒ_dpIn_raw + ℒ_Σx_raw + ℒ_Σy_raw + ℒ_cont_raw + ℒ_mass_raw + ℒ_ϕBC_raw
        ℒ_individuals = [ℒ_J, ℒ_Jwall, ℒ_uWall, ℒ_dpIn, ℒ_Σx, ℒ_Σy, ℒ_cont, ℒ_mass, ℒ_ϕBC]
        ℒ_individuals_no_weights = [ℒ_J_raw, ℒ_Jwall_raw, ℒ_uWall_raw, ℒ_dpIn_raw, ℒ_Σx_raw, ℒ_Σy_raw, ℒ_cont_raw, ℒ_mass_raw, ℒ_ϕBC_raw]
    else:
        ℒ = ℒ_J + ℒ_Jwall + ℒ_uWall + ℒ_dpIn + ℒ_Σx + ℒ_Σy + ℒ_cont + ℒ_mass
        ℒ_no_weights = ℒ_J_raw + ℒ_Jwall_raw + ℒ_uWall_raw + ℒ_dpIn_raw + ℒ_Σx_raw + ℒ_Σy_raw + ℒ_cont_raw + ℒ_mass_raw
        ℒ_individuals = [ℒ_J, ℒ_Jwall, ℒ_uWall, ℒ_dpIn, ℒ_Σx, ℒ_Σy, ℒ_cont, ℒ_mass]
        ℒ_individuals_no_weights = [ℒ_J_raw, ℒ_Jwall_raw, ℒ_uWall_raw, ℒ_dpIn_raw, ℒ_Σx_raw, ℒ_Σy_raw, ℒ_cont_raw, ℒ_mass_raw]
    # -
    return ℒ, ℒ_no_weights, ℒ_individuals, ℒ_individuals_no_weights
