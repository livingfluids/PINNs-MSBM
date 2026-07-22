import torch
import loss_schemes 
import config

# NOTE: Normalized values contain '*_' suffix

EPS = 1e-12


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

# Calculate Gradient Helper Function
def grad(y, f):
    y = y if y.ndim == 2 else y.unsqueeze(1)
    df_dy = torch.autograd.grad(f, y, torch.ones_like(f), create_graph=True)[0]  # torch.Size([y, 1])
    return df_dy

# u Data Loss Term   
def uDataLoss(trials, params): 
    if config.PROBLEM == 'inverse': return trials.u_trial(params.y_data_) - params.u_data_
    elif config.PROBLEM == 'forward': 
        y_ = params.y_coll_
        u = trials.u_trial(y_)
        return u[:1] + u[-1:]

# ϕ Mean Squared Error Ray[Tune] Metric
def ϕMSE(trials, params): return torch.mean((trials.ϕ_trial(params.y_data_) - params.ϕ_data_)**2)

# MSBM Equations Loss Terms | ∇⋅J = ∇⋅Σ = 0
def PDELoss(trials, params, device):
    # Initialize 
    y_ = params.y_coll_        # already normalized 
    u_ = trials.u_trial(y_)    # already normalized 
    ϕ_ = trials.ϕ_trial(y_)    # ϕ itself is not normalized, but we do calculate it here along y_
    A = 2 * params.a / params.H
    p_ = params.p * params.H / (2 * params.η0 * params.u_max)
    zero = torch.zeros_like(y_, device=device)  # torch.Size([y, 1])

    # Normal stress viscosity (ηₙ(ϕ))
    def ηN(ϕ): return params.Kn * (ϕ/params.ϕ_max)**2 * (1 - ϕ/params.ϕ_max)**(-2)  # torch.Size([y_.shape[0], 1]), a scalar for each y

    # Shear viscosity of the particle phase (ηₚ(ϕ))
    def ηp(ϕ):
        ηs = (1 - ϕ/params.ϕ_max)**(-2)
        return ηs - 1  # torch.Size([y, 1]), a scalar for each y

    # Sedimentation hinderence function for mobility of particle phase (f(ϕ))
    def f(ϕ): return (1 - ϕ/params.ϕ_max) * (1 - ϕ)**(params.α - 1)  # torch.Size([y, 1]), a scalar for each y

    # Gradient of the velocity field (∇U)
    du_dy_ = grad(y_, u_)  # torch.Size([y, 1])
    U_grad_ = torch.stack([
        torch.cat([zero, du_dy_, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Strain rate tensor (E)
    E_ = 0.5 * (U_grad_ + U_grad_.transpose(1, 2))  # torch.Size([y, 3, 3]), a matrix for each y

    # Shear rate tensor (γ̇)
    γ̇_ = torch.sqrt(2 * torch.sum(E_ * E_, dim=(1, 2))).unsqueeze(1)  # torch.Size([y, 1])

    # Lift force (L)
    γ̇ = γ̇_ * 2 * params.u_max / params.H  # dimensionalize for calculating it
    left_wall = 3 * params.η0 * γ̇ / (4 * torch.pi * ((params.H/2)*(y_ + 1) + params.H0)**params.β) * params.frv
    right_wall = 3 * params.η0 * γ̇ / (4 * torch.pi * ((params.H/2)*(1 - y_) + params.H0)**params.β) * params.frv
    scale_L = (params.H ** 2) / (2 * params.η0 * params.u_max)  # nondimensionalize after calculating it 
    L = torch.stack([
        torch.cat([zero], dim=1),
        torch.cat([scale_L * (left_wall - right_wall)], dim=1),
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
    dΣpxy_dy_ = grad(y_, Σp_[:, 0, 1])  # torch.Size([y, 1])
    dΣpyy_dy_ = grad(y_, Σp_[:, 1, 1])  # torch.Size([y, 1])
    Σp_div_ = torch.stack([
        torch.cat([zero + dΣpxy_dy_ + zero], dim=1),
        torch.cat([zero + dΣpyy_dy_ + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y

    # Migration flux (J)
    J_ = - (2 * A**2 / 9) * f(ϕ_).unsqueeze(1) * (Σp_div_ + ϕ_.view(-1, 1, 1) * L)  # torch.Size([y, 3, 1])

    # Soft enforce zero migration flux at walls
    Jy_wall_ = torch.stack([J_[0,1,0], J_[-1,1,0]])

    # Divergence of migration flux (∇⋅J)
    dJx_dx_ = dJz_dz_ = zero
    dJy_dy_ = grad(y_, J_[:, 1, 0])  # torch.Size([y, 1])
    J_div_ = dJx_dx_ + dJy_dy_ + dJz_dz_  # torch.Size([y, 1])

    # Identity matrix (I)
    I = torch.eye(3, device=device).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid phase stress (Σᶠ)
    Σf_ = - p_ * I + 2 * E_

    # Total stress (Σ)
    Σ_ = Σp_ + Σf_

    # Suspension momentum balance (∇⋅Σ)
    dΣxy_dy_ = grad(y_, Σ_[:, 0, 1])  # torch.Size([y, 1])
    dΣyy_dy_ = grad(y_, Σ_[:, 1, 1])  # torch.Size([y, 1])

    # Add pressure gradient
    dΣxy_dy_ -= params.dp_dx_

    return J_div_, Jy_wall_, dΣxy_dy_, dΣyy_dy_

# Velocity-Weighted Bulk Conservation Loss Term
def ϕBulkLoss(trials, params):
    y_ = params.y_coll_
    u_ = trials.u_trial(y_) 
    ϕ  = trials.ϕ_trial(y_)
    ϕ_bulk_term = torch.sum(ϕ * u_) / torch.sum(u_) - params.ϕ_bulk
    return ϕ_bulk_term

# CFL Term (for learnable beta only)
def ϕBCLoss(trials, params):
    y_ = params.y_coll_
    ϕ = trials.ϕ_trial(y_)
    if config.PROBLEM == 'inverse': return ϕ[:params.CFL_] + ϕ[-params.CFL_:]
    elif config.PROBLEM == 'forward': return ϕ[:1] + ϕ[-1]

# u & ϕ Symmetry Constraint Loss Term
def symmetryLoss(trials, params):  # ensures ϕ is symmetric along centerflow axis
    y_ = params.y_coll_
    u_sym_term = trials.u_trial(y_) - trials.u_trial(-y_)
    ϕ_sym_term = trials.ϕ_trial(y_) - trials.ϕ_trial(-y_)
    # - 
    # u = trials.u_trial(y_)
    # u_sym_term += u[:1] + u[-1:]
    return u_sym_term, ϕ_sym_term

# Safe Ratio Helper Function
def safeRatio(num, den):
    return num / (den + EPS)

def clampAdaptiveWeight(weight):
    #weight = torch.clamp(weight, min=0, max=1)
    #weight = weight**2 * (3 - 2 * weight) * config.ADAPTIVE_WEIGHT_MAX
    return weight
    # return torch.clamp(weight, min=ADAPTIVE_WEIGHT_MIN, max=ADAPTIVE_WEIGHT_MAX)

# Max Gradient Helper Function
def maxGradMagnitude(loss, PINN, params):
    param_list = list(PINN.parameters()) + [params.β] if config.CASE == 'learn_beta' else list(PINN.parameters())
    grads = torch.autograd.grad(loss, param_list, retain_graph=True, create_graph=False, allow_unused=True)
    max_abs = 0.0
    for g in grads:
        if g is not None:
            gmax = torch.max(torch.abs(g))
            if torch.isfinite(gmax): max_abs = max(max_abs, gmax.item())
    return torch.tensor(max_abs, device=loss.device) + EPS

# Mean Absolute Gradient Helper Function
def meanAbsGrad(loss, PINN, params):
    param_list = list(PINN.parameters()) + [params.β] if config.CASE == 'learn_beta' else list(PINN.parameters())
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
    include_ϕ_BC = (config.CASE == 'learn_beta' or config.PROBLEM == 'forward') 
    # -     
    u_data_term                             = uDataLoss(trials, params)
    J_div_, Jy_wall_, dΣxy_dy_, dΣyy_dy_    = PDELoss(trials, params, device)
    ϕ_bulk_term                             = ϕBulkLoss(trials, params)
    u_sym_term, ϕ_sym_term                  = symmetryLoss(trials, params)
    # - 
    ℒ_data_raw      = torch.mean(params.mask(params.λ_data)     * u_data_term ** 2)
    ℒ_J_raw         = torch.mean(params.mask(params.λ_J)        * J_div_ ** 2)
    ℒ_Jy_raw        = torch.mean(params.mask(params.λ_Jy_wall)  * Jy_wall_ ** 2)
    ℒ_Σxy_raw       = torch.mean(params.mask(params.λ_Σxy)      * dΣxy_dy_ ** 2)
    ℒ_Σyy_raw       = torch.mean(params.mask(params.λ_Σyy)      * dΣyy_dy_ ** 2)
    ℒ_mass_raw      = torch.mean(params.mask(params.λ_mass)     * ϕ_bulk_term ** 2)
    ℒ_u_raw         = torch.mean(params.mask(params.λ_u_sym)    * u_sym_term ** 2)
    ℒ_ϕ_raw         = torch.mean(params.mask(params.λ_ϕ_sym)    * ϕ_sym_term ** 2)
    # True unweighted component losses for reporting/plotting only.
    ℒ_data_true     = torch.mean(u_data_term ** 2)
    ℒ_J_true        = torch.mean(J_div_ ** 2)
    ℒ_Jy_true       = torch.mean(Jy_wall_ ** 2)
    ℒ_Σxy_true      = torch.mean(dΣxy_dy_ ** 2)
    ℒ_Σyy_true      = torch.mean(dΣyy_dy_ ** 2)
    ℒ_mass_true     = torch.mean(ϕ_bulk_term ** 2)
    ℒ_u_true        = torch.mean(u_sym_term ** 2)
    ℒ_ϕ_true        = torch.mean(ϕ_sym_term ** 2)
    if include_ϕ_BC:
        ϕ_BC_term = ϕBCLoss(trials, params)
        ℒ_BC_raw = torch.mean(params.mask(params.λ_BC) * ϕ_BC_term ** 2)
        ℒ_BC_true = torch.mean(ϕ_BC_term ** 2)
    else:
        ℒ_BC_raw = torch.zeros((), device=device, dtype=ℒ_J_raw.dtype)
        ℒ_BC_true = torch.zeros((), device=device, dtype=ℒ_J_true.dtype)
    # - 
    if epoch % config.GRAD_NORM_EPOCH_INTERVAL == 0:
        g_J = maxGradMagnitude(ℒ_J_raw, PINN, params)
        g_Σxy = maxGradMagnitude(ℒ_Σxy_raw, PINN, params)
        g_Σyy = maxGradMagnitude(ℒ_Σyy_raw, PINN, params)
        g_res_max = torch.max(torch.stack([g_J, g_Σxy, g_Σyy]))
        # - 
        g_data = meanAbsGrad(ℒ_data_raw, PINN, params)
        g_Jy = meanAbsGrad(ℒ_Jy_raw, PINN, params)
        g_mass = meanAbsGrad(ℒ_mass_raw, PINN, params)
        g_u = meanAbsGrad(ℒ_u_raw, PINN, params)
        g_ϕ = meanAbsGrad(ℒ_ϕ_raw, PINN, params)
        if include_ϕ_BC: g_BC = meanAbsGrad(ℒ_BC_raw, PINN, params)
        # - 
        Λ_data = clampAdaptiveWeight(safeRatio(g_res_max, g_data))
        Λ_J = clampAdaptiveWeight(safeRatio(g_res_max, g_J))
        Λ_Jy = clampAdaptiveWeight(safeRatio(g_res_max, g_Jy))
        Λ_Σxy = clampAdaptiveWeight(safeRatio(g_res_max, g_Σxy))
        Λ_Σyy = clampAdaptiveWeight(safeRatio(g_res_max, g_Σyy))
        Λ_mass = clampAdaptiveWeight(safeRatio(g_res_max, g_mass))
        Λ_u = clampAdaptiveWeight(safeRatio(g_res_max, g_u))
        Λ_ϕ = clampAdaptiveWeight(safeRatio(g_res_max, g_ϕ))
        if include_ϕ_BC: Λ_BC = clampAdaptiveWeight(safeRatio(g_res_max, g_BC))
        # - 
        ξ = config.ξ
        Λ_data_new = ξ * Λ_data_old + (1 - ξ) * Λ_data
        Λ_J_new = ξ * Λ_J_old + (1 - ξ) * Λ_J
        Λ_Jy_new = ξ * Λ_Jy_old + (1 - ξ) * Λ_Jy
        Λ_Σxy_new = ξ * Λ_Σxy_old + (1 - ξ) * Λ_Σxy
        Λ_Σyy_new = ξ * Λ_Σyy_old + (1 - ξ) * Λ_Σyy
        Λ_mass_new = ξ * Λ_mass_old + (1 - ξ) * Λ_mass
        Λ_u_new = ξ * Λ_u_sym_old + (1 - ξ) * Λ_u
        Λ_ϕ_new = ξ * Λ_ϕ_sym_old + (1 - ξ) * Λ_ϕ
        if include_ϕ_BC: Λ_BC_new = ξ * Λ_BC_old + (1 - ξ) * Λ_BC
        # - 
        Λ_data_old = Λ_data_new
        Λ_J_old, Λ_Jy_old, Λ_Σxy_old, Λ_Σyy_old = Λ_J_new, Λ_Jy_new, Λ_Σxy_new, Λ_Σyy_new
        Λ_mass_old, Λ_u_sym_old, Λ_ϕ_sym_old = Λ_mass_new, Λ_u_new, Λ_ϕ_new
        if include_ϕ_BC: Λ_BC_old = Λ_BC_new
        else: Λ_BC_old = 1.0

    ℒ_data = (ℒ_data_raw * config.Λ_data) * Λ_data_old
    ℒ_J = (ℒ_J_raw * config.Λ_PDEs) * Λ_J_old
    ℒ_Jy = (ℒ_Jy_raw * config.Λ_BCs) * Λ_Jy_old
    ℒ_Σxy = (ℒ_Σxy_raw * config.Λ_PDEs) * Λ_Σxy_old
    ℒ_Σyy = (ℒ_Σyy_raw * config.Λ_PDEs) * Λ_Σyy_old
    ℒ_mass = (ℒ_mass_raw * config.Λ_BCs) * Λ_mass_old
    ℒ_u = (ℒ_u_raw * config.Λ_BCs) * Λ_u_sym_old
    ℒ_ϕ = (ℒ_ϕ_raw * config.Λ_BCs) * Λ_ϕ_sym_old
    if include_ϕ_BC: ℒ_BC = (ℒ_BC_raw * config.Λ_BCs) * Λ_BC_old
    else: ℒ_BC = torch.zeros((), device=device, dtype=ℒ_J.dtype)
    # - 
    if include_ϕ_BC:
        ℒ = ℒ_J + ℒ_Jy + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_BC + ℒ_u + ℒ_ϕ + ℒ_data
        ℒ_no_weights = ℒ_J_true + ℒ_Jy_true + ℒ_Σxy_true + ℒ_Σyy_true + ℒ_mass_true + ℒ_BC_true + ℒ_u_true + ℒ_ϕ_true + ℒ_data_true
        ℒ_individuals = [ℒ_J, ℒ_Jy, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_BC, ℒ_u, ℒ_ϕ, ℒ_data]
        ℒ_individuals_no_weights = [ℒ_J_true, ℒ_Jy_true, ℒ_Σxy_true, ℒ_Σyy_true, ℒ_mass_true, ℒ_BC_true, ℒ_u_true, ℒ_ϕ_true, ℒ_data_true]
    
    elif (not include_ϕ_BC):
        ℒ = ℒ_J + ℒ_Jy + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_u + ℒ_ϕ + ℒ_data
        ℒ_no_weights = ℒ_J_true + ℒ_Jy_true + ℒ_Σxy_true + ℒ_Σyy_true + ℒ_mass_true + ℒ_u_true + ℒ_ϕ_true + ℒ_data_true
        ℒ_individuals = [ℒ_J, ℒ_Jy, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_u, ℒ_ϕ, ℒ_data]
        ℒ_individuals_no_weights = [ℒ_J_true, ℒ_Jy_true, ℒ_Σxy_true, ℒ_Σyy_true, ℒ_mass_true, ℒ_u_true, ℒ_ϕ_true, ℒ_data_true]

    return ℒ, ℒ_no_weights, ℒ_individuals, ℒ_individuals_no_weights
