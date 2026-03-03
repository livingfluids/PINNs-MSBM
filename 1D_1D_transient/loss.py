import torch
import config

# NOTE: Normalized values contain '*_' suffix

EPS = 1e-12
PHI_FRAC_EPS = getattr(config, "PHI_FRAC_EPS", 1e-6)
ADAPTIVE_WEIGHT_MIN = getattr(config, "ADAPTIVE_WEIGHT_MIN", 1e-3)
ADAPTIVE_WEIGHT_MAX = getattr(config, "ADAPTIVE_WEIGHT_MAX", 1e3)

# Cached adaptive multipliers for grad-norm scheme
Λ_J_old = 1.0
Λ_Jy_old = 1.0
Λ_Σxy_old = 1.0
Λ_Σyy_old = 1.0
Λ_mass_old = 1.0
Λ_BC_old = 1.0
Λ_u_sym_old = 1.0
Λ_ϕ_sym_old = 1.0
Λ_data_old = 1.0

# Gradient Helper Function
def grad(y, f):
    y = y if y.ndim == 2 else y.unsqueeze(1)
    return torch.autograd.grad(f, y, torch.ones_like(f), create_graph=True)[0]

# u Data Loss
def uDataLoss(trials, params):
    u_pred, _ = trials.u_trial(params.y_data_, params.t_data_)
    u_true = params.u_data_.permute(1, 0, 2)
    return u_pred - u_true

# ϕ MSE (Not Used For Loss)
def ϕMSE(trials, params):
    ϕ_pred, _ = trials.ϕ_trial(params.y_data_, params.t_data_)
    ϕ_true = params.ϕ_data_.permute(1, 0, 2)
    return torch.mean((ϕ_pred - ϕ_true) ** 2)

# PDE Loss
def PDELoss(trials, params, device, PINN):
    y_ = params.y_flat_.clone().detach().requires_grad_(True)
    τ_ = params.t_flat_.clone().detach().requires_grad_(True)
    # - 
    input_flat = torch.cat([y_, τ_], dim=1)
    u_ϕ_ = PINN(input_flat)
    # - 
    u_ = torch.sigmoid(u_ϕ_[:, 0:1])
    ϕ_ = params.ϕ_max * torch.sigmoid(u_ϕ_[:, 1:2])
    # - 
    A = 2 * params.a / params.H
    p_ = params.p * params.H / (2 * params.η0 * params.u_max)
    # - 
    zero = torch.zeros_like(u_, device=device)

    # Normal stress viscosity (ηₙ(ϕ))
    def ηN(ϕ_): return params.Kn * (ϕ_ / params.ϕ_max)**2 * (1.0 - (ϕ_ / params.ϕ_max))**(-2)  # [y·t, 1]

    # Shear viscosity of the particle phase (ηₚ(ϕ))
    def ηp(ϕ_):
        ηs = (1.0 - (ϕ_ / params.ϕ_max))**(-2)
        return ηs - 1  # [y·t, 1]

    # Sedimentation hinderence function for mobility of particle phase (f(ϕ))
    def f(ϕ_): return (1.0 - (ϕ_ / params.ϕ_max)) * (1.0 - ϕ_)**(params.α - 1)  # [y·t, 1]

    # Gradient of the velocity field (∇U)
    du_dy_ = grad(y_, u_)  # torch.Size([y, 1])
    U_grad_ = torch.stack([
        torch.cat([zero, du_dy_, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1)
    ], dim=1)  # [y·t, 3, 3], a matrix for each y·t

    # Strain rate tensor (E)
    E_ = 0.5 * (U_grad_ + U_grad_.transpose(1, 2))  # [y·t, 3, 3], a matrix for each y·t

    # Shear rate tensor (γ̇)
    γ̇_ = torch.sqrt(2 * torch.sum(E_ * E_, dim=(1, 2))).unsqueeze(1)  # [y·t, 1]

    # Lift force (L)
    γ̇ = γ̇_ * 2 * params.u_max / params.H  # dimensionalize for calculating it
    left_wall = 3 * params.η0 * γ̇ / (4 * torch.pi * ((params.H/2)*(y_ + 1) + params.H0)**params.β) * params.frv
    right_wall = 3 * params.η0 * γ̇ / (4 * torch.pi * ((params.H/2)*(1 - y_) + params.H0)**params.β) * params.frv
    scale_L = (params.H ** 2) / (2 * params.η0 * params.u_max)  # nondimensionalize after calculating it 
    L = torch.stack([
        torch.cat([zero], dim=-1),  # [y⋅t, 1]
        torch.cat([scale_L * (left_wall - right_wall)], dim=-1),  # [y⋅t, 1]
        torch.cat([zero], dim=-1)   # [y⋅t, 1]
    ], dim=-2)  # [y·t, 3, 1], a vector for each y·t
    
    # Diagonal tensor of the SBM (Q)
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, params.λ2, 0.0], [0.0, 0.0, params.λ3]], device=device).repeat(y_.shape[0], 1, 1)  # [y⋅t, 3, 3], a matrix for each y·t

    # Non-local shear rate tensor
    γ̇NL_ = params.ε * params.H / 2

    # Particle normal stress diagonal tensor (Σₙₙᵖ)
    Σpnn_ = ηN(ϕ_).view(-1, 1, 1) * (γ̇_.unsqueeze(1) + γ̇NL_) * Q  # [y·t, 3, 3], a matrix for each y·t

    # Oriented particle stress tensor (Σᵖ)
    Σp_ = -Σpnn_ + (2 * ηp(ϕ_).view(-1, 1, 1) * E_)  # [y·t, 3, 3], a matrix for each (y, t)

    # Divergence of oriented particle stress tensor (∇⋅Σᵖ)
    dΣpxy_dy_ = grad(y_, Σp_[:, 0, 1].reshape(-1, 1))  # torch.Size([y, 1])
    dΣpyy_dy_ = grad(y_, Σp_[:, 1, 1].reshape(-1, 1))  # torch.Size([y, 1])
    Σp_div_ = torch.stack([
        torch.cat([zero + dΣpxy_dy_ + zero], dim=1),
        torch.cat([zero + dΣpyy_dy_ + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
    ], dim=1)  # [y⋅t, 3, 1], a vector for each y⋅t 

    # Migration flux (J)
    J_ = - (2 * A**2 / 9) * f(ϕ_).unsqueeze(1) * (Σp_div_ + ϕ_.view(-1, 1, 1) * L)  # [y⋅t, 3, 1], a vector for each y⋅t

    # Soft enforce zero migration flux at walls
    J_grid_ = J_.reshape(config.y_COLL, config.t_COLL, 3, 1)
    Jy_wall_ = torch.stack([J_grid_[0, :, 1, 0], J_grid_[-1, :, 1, 0]])  # [2, t], a tuple for each t


    # Divergence of migration flux (∇⋅J)
    dJx_dx_ = dJz_dz_ = zero
    dJy_dy_ = grad(y_, J_[:, 1, 0].reshape(-1, 1))  # [y⋅t, 1]
    J_div_ = dJx_dx_ + dJy_dy_ + dJz_dz_  # [y⋅t, 1]

    # Identity matrix (I)
    I = torch.eye(3, device=device).repeat(y_.shape[0], 1, 1)  # [y⋅t, 3, 3], a matrix for each y⋅t

    # Fluid phase stress (Σᶠ)
    Σf_ = - p_ * I + 2 * E_  # [y·t, 3, 3], a matrix for each y⋅t

    # Total stress (Σ)
    Σ_ = Σp_ + Σf_  # [y·t, 3, 3], a matrix for each y⋅t

    # Suspension momentum balance (∇⋅Σ)
    dΣxy_dy_ = grad(y_, Σ_[:, 0, 1].reshape(-1, 1))  # [y⋅t, 1]
    dΣyy_dy_ = grad(y_, Σ_[:, 1, 1].reshape(-1, 1))  # [y⋅t, 1]

    # Time derivatives w.r.t log time (τ)
    dϕ_dτ_ = grad(τ_, ϕ_)  # [y⋅t, 1]
    du_dτ_ = grad(τ_, u_)  # [y⋅t, 1]

    # Chain rule: ∂/∂t_ = T_ref * (1 / (dt/dτ)) * ∂/∂τ
    t = torch.exp(τ_ * params.t_log_span + params.t_log_min) - params.t_eps
    dt_dτ = (t + params.t_eps) * params.t_log_span
    T_ref = params.H**2 / (4 * params.η0)

    dϕ_dt_ = dϕ_dτ_ / (dt_dτ + EPS) * T_ref
    du_dt_ = du_dτ_ / (dt_dτ + EPS) * T_ref

    # Convective term
    # u_grad_ϕ = grad(y_, u_ * ϕ_)

    # Compute Residuals 
    J_term = J_div_ + dϕ_dt_ # + u_grad_ϕ
    Jy_wall_term = Jy_wall_
    dΣxy_term = dΣxy_dy_ - params.dp_dx_ - du_dt_
    dΣyy_term = dΣyy_dy_
    # - 
    return J_term, Jy_wall_term, dΣxy_term, dΣyy_term

# Bulk ϕ Loss
def ϕBulkLoss(trials, params):
    u_grid, _ = trials.u_trial(params.y_coll_, params.t_coll_)
    ϕ_grid, _ = trials.ϕ_trial(params.y_coll_, params.t_coll_)
    # - 
    u_grid = u_grid.squeeze(-1)
    ϕ_grid = ϕ_grid.squeeze(-1)
    # - 
    weighted_avg = torch.sum(ϕ_grid * u_grid, dim=0) / (torch.sum(u_grid, dim=0) + EPS)
    # - 
    return weighted_avg - params.ϕ_bulk

# ϕ BCs Loss (ϕ at t=0 & CFL at t=T)
def ϕBCLoss(trials, params):
    t_coll = params.t_coll_
    y_coll = params.y_coll_
    # - 
    t_zero = t_coll[0:1]
    ϕ_at_t0, _ = trials.ϕ_trial(y_coll, t_zero)
    residual_ic = (ϕ_at_t0.squeeze(-1).squeeze(-1) - params.ϕ_bulk).reshape(-1)
    # - 
    cfl = getattr(params, "CFL_", 1)
    cfl = int(cfl.item()) if torch.is_tensor(cfl) else int(cfl)
    n_bc = max(1, min(cfl, y_coll.shape[0] // 2))
    t_final = t_coll[-1:]
    # - 
    y_left = y_coll[:n_bc]
    y_right = y_coll[-n_bc:]
    ϕ_left, _ = trials.ϕ_trial(y_left, t_final)
    ϕ_right, _ = trials.ϕ_trial(y_right, t_final)
    # - 
    residual_bc_left = ϕ_left.reshape(-1)
    residual_bc_right = ϕ_right.reshape(-1)
    return torch.cat([residual_ic, residual_bc_left, residual_bc_right], dim=0)

# Symmetry Loss (u & ϕ)
def symmetryLoss(trials, params):
    y_pos = params.y_coll_
    y_neg = -params.y_coll_
    t_ = params.t_coll_
    # - 
    u_pos, _ = trials.u_trial(y_pos, t_)
    u_neg, _ = trials.u_trial(y_neg, t_)
    ϕ_pos, _ = trials.ϕ_trial(y_pos, t_)
    ϕ_neg, _ = trials.ϕ_trial(y_neg, t_)
    # - 
    u_sym_term = (u_pos - u_neg).permute(1, 0, 2)
    ϕ_sym_term = (ϕ_pos - ϕ_neg).permute(1, 0, 2)
    # - 
    return u_sym_term, ϕ_sym_term

# Safe Ratio Helper Function
def safeRatio(num, den):
    return num / (den + EPS)

def clampAdaptiveWeight(weight):
    return torch.clamp(weight, min=ADAPTIVE_WEIGHT_MIN, max=ADAPTIVE_WEIGHT_MAX)

# Max Gradient Helper Function
def maxGradMagnitude(loss, PINN, params):
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

def lossGradientNormalizationMax(trials, params, PINN, epoch, device):
    global Λ_J_old, Λ_Jy_old, Λ_Σxy_old, Λ_Σyy_old, Λ_mass_old, Λ_BC_old, Λ_u_sym_old, Λ_ϕ_sym_old, Λ_data_old
    # - 
    J_div_, Jy_wall_, dΣxy_dy_, dΣyy_dy_    = PDELoss(trials, params, device, PINN)
    ϕ_bulk_term                             = ϕBulkLoss(trials, params)
    ϕ_BC_term                               = ϕBCLoss(trials, params)
    u_sym_term, ϕ_sym_term                  = symmetryLoss(trials, params)
    u_data_term                             = uDataLoss(trials, params)
    # - 
    ℒ_J_raw         = torch.mean(params.mask(params.λ_J)        * J_div_ ** 2)
    ℒ_Jy_raw        = torch.mean(params.mask(params.λ_Jy_wall)  * Jy_wall_ ** 2)
    ℒ_Σxy_raw       = torch.mean(params.mask(params.λ_Σxy)      * dΣxy_dy_ ** 2)
    ℒ_Σyy_raw       = torch.mean(params.mask(params.λ_Σyy)      * dΣyy_dy_ ** 2)
    ℒ_mass_raw      = torch.mean(params.mask(params.λ_mass)     * ϕ_bulk_term ** 2)
    ℒ_BC_raw        = torch.mean(params.mask(params.λ_BC)       * ϕ_BC_term ** 2)
    ℒ_u_raw         = torch.mean(params.mask(params.λ_u_sym)    * u_sym_term ** 2)
    ℒ_ϕ_raw         = torch.mean(params.mask(params.λ_ϕ_sym)    * ϕ_sym_term ** 2)
    ℒ_data_raw      = torch.mean(params.mask(params.λ_data)     * u_data_term ** 2)
    # - 
    if epoch % config.GRAD_NORM_EPOCH_INTERVAL == 0:
        g_J = maxGradMagnitude(ℒ_J_raw, PINN, params)
        g_Σxy = maxGradMagnitude(ℒ_Σxy_raw, PINN, params)
        g_Σyy = maxGradMagnitude(ℒ_Σyy_raw, PINN, params)
        g_res_max = torch.max(torch.stack([g_J, g_Σxy, g_Σyy]))
        # - 
        g_Jy = meanAbsGrad(ℒ_Jy_raw, PINN, params)
        g_mass = meanAbsGrad(ℒ_mass_raw, PINN, params)
        g_BC = meanAbsGrad(ℒ_BC_raw, PINN, params)
        g_u = meanAbsGrad(ℒ_u_raw, PINN, params)
        g_ϕ = meanAbsGrad(ℒ_ϕ_raw, PINN, params)
        g_data = meanAbsGrad(ℒ_data_raw, PINN, params)
        # - 
        Λ_J = clampAdaptiveWeight(safeRatio(g_res_max, g_J))
        Λ_Jy = clampAdaptiveWeight(safeRatio(g_res_max, g_Jy))
        Λ_Σxy = clampAdaptiveWeight(safeRatio(g_res_max, g_Σxy))
        Λ_Σyy = clampAdaptiveWeight(safeRatio(g_res_max, g_Σyy))
        Λ_mass = clampAdaptiveWeight(safeRatio(g_res_max, g_mass))
        Λ_BC = clampAdaptiveWeight(safeRatio(g_res_max, g_BC))
        Λ_u = clampAdaptiveWeight(safeRatio(g_res_max, g_u))
        Λ_ϕ = clampAdaptiveWeight(safeRatio(g_res_max, g_ϕ))
        Λ_data = clampAdaptiveWeight(safeRatio(g_res_max, g_data))
        # - 
        ξ = config.ξ
        Λ_J_new = ξ * Λ_J_old + (1 - ξ) * Λ_J
        Λ_Jy_new = ξ * Λ_Jy_old + (1 - ξ) * Λ_Jy
        Λ_Σxy_new = ξ * Λ_Σxy_old + (1 - ξ) * Λ_Σxy
        Λ_Σyy_new = ξ * Λ_Σyy_old + (1 - ξ) * Λ_Σyy
        Λ_mass_new = ξ * Λ_mass_old + (1 - ξ) * Λ_mass
        Λ_BC_new = ξ * Λ_BC_old + (1 - ξ) * Λ_BC
        Λ_u_new = ξ * Λ_u_sym_old + (1 - ξ) * Λ_u
        Λ_ϕ_new = ξ * Λ_ϕ_sym_old + (1 - ξ) * Λ_ϕ
        Λ_data_new = ξ * Λ_data_old + (1 - ξ) * Λ_data
        # - 
        Λ_J_old, Λ_Jy_old, Λ_Σxy_old, Λ_Σyy_old = Λ_J_new, Λ_Jy_new, Λ_Σxy_new, Λ_Σyy_new
        Λ_mass_old, Λ_BC_old, Λ_u_sym_old, Λ_ϕ_sym_old, Λ_data_old = (
            Λ_mass_new,
            Λ_BC_new,
            Λ_u_new,
            Λ_ϕ_new,
            Λ_data_new,
        )

    ℒ_J = (ℒ_J_raw * config.Λ_PDEs) * Λ_J_old
    ℒ_Jy = (ℒ_Jy_raw * config.Λ_BCs) * Λ_Jy_old
    ℒ_Σxy = (ℒ_Σxy_raw * config.Λ_PDEs) * Λ_Σxy_old
    ℒ_Σyy = (ℒ_Σyy_raw * config.Λ_PDEs) * Λ_Σyy_old
    ℒ_mass = (ℒ_mass_raw * config.Λ_BCs) * Λ_mass_old
    ℒ_BC = (ℒ_BC_raw * config.Λ_BCs) * Λ_BC_old
    ℒ_u = (ℒ_u_raw * config.Λ_BCs) * Λ_u_sym_old
    ℒ_ϕ = (ℒ_ϕ_raw * config.Λ_BCs) * Λ_ϕ_sym_old
    ℒ_data = (ℒ_data_raw * config.Λ_data) * Λ_data_old
    # - 
    ℒ = ℒ_J + ℒ_Jy + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_BC + ℒ_u + ℒ_ϕ + ℒ_data
    ℒ_individuals = [ℒ_J, ℒ_Jy, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_BC, ℒ_u, ℒ_ϕ, ℒ_data]
    # - 
    return ℒ, ℒ_individuals
