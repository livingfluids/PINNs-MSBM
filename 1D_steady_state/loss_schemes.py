import torch
import config

# Script Constants
Λ_J_old = Λ_Jy_old = Λ_Σxy_old = Λ_Σyy_old = Λ_mass_old = Λ_BC_old = Λ_u_sym_old = Λ_ϕ_sym_old = Λ_data_old = 1
ℒ_J_init, ℒ_Jy_init, ℒ_Σxy_init, ℒ_Σyy_init, ℒ_mass_init, ℒ_BC_init, ℒ_u_sym_init, ℒ_ϕ_sym_init, ℒ_data_init, ℒ_init = None, None, None, None, None, None, None, None, None, None
EPS = 1e-12

# Safe Ratio Helper Function
def safeRatio(num, den): return num / (den + EPS)

# Calculate Gradient Norm Helper Function
def gradientNormalize(loss, PINN, params):
    if config.CASE == 'learn_beta': param_list = list(PINN.parameters()) + [params.β]
    elif config.CASE == 'learn_cfl': param_list = list(PINN.parameters())
    else: raise ValueError(f"Invalid CASE: {config.CASE}")
    grads = torch.autograd.grad(loss, param_list, retain_graph=True, create_graph=False, allow_unused=True)
    sq = 0.0
    for g in grads:
        if g is None: continue
        else: sq = sq + g.detach().pow(2).sum()
    return torch.sqrt(sq + EPS)

# Max Gradient Norm Element
def maxGradMagnitude(loss, PINN, params):
    if config.CASE == 'learn_beta': param_list = list(PINN.parameters()) + [params.β]
    else: param_list = list(PINN.parameters())
    grads = torch.autograd.grad(loss, param_list, retain_graph=True, create_graph=False, allow_unused=True)
    max_abs = 0.0
    for g in grads:
        if g is not None:
            max_abs = max(max_abs, torch.max(torch.abs(g)).item())
    return torch.tensor(max_abs, device=loss.device) + EPS  # avoid zero

def meanAbsGrad(loss, PINN, params):
    if config.CASE == 'learn_beta': param_list = list(PINN.parameters()) + [params.β]
    else: param_list = list(PINN.parameters())
    grads = torch.autograd.grad(loss, param_list, retain_graph=True, create_graph=False, allow_unused=True)
    total_abs = 0.0
    count = 0
    for g in grads:
        if g is not None:
            total_abs += torch.sum(torch.abs(g))
            count += g.numel()
    return torch.tensor(total_abs / count if count > 0 else EPS, device=loss.device)

# Gradient Norm Loss Scheme
def lossGradientNormalization(loss_dict, params, PINN, epoch):
    global Λ_J_old, Λ_Jy_old, Λ_Σxy_old, Λ_Σyy_old, Λ_mass_old, Λ_BC_old, Λ_u_sym_old, Λ_ϕ_sym_old, Λ_data_old

    # --- define *raw* component losses (no config.Λ_* yet) ---
    ℒ_J_raw     = torch.mean(params.mask(params.λ_J)        * loss_dict['J_div_']**2)
    ℒ_Jy_raw    = torch.mean(params.mask(params.λ_Jy_wall)  * loss_dict['Jy_wall_']**2)
    ℒ_Σxy_raw   = torch.mean(params.mask(params.λ_Σxy)      * loss_dict['dΣxy_dy_']**2)
    ℒ_Σyy_raw   = torch.mean(params.mask(params.λ_Σyy)      * loss_dict['dΣyy_dy_']**2)
    ℒ_mass_raw  = torch.mean(params.mask(params.λ_mass)     * loss_dict['ϕ_bulk_term']**2)
    ℒ_BC_raw    = torch.mean(params.mask(params.λ_BC)       * loss_dict['ϕ_BC_term']**2)
    ℒ_u_raw     = torch.mean(params.mask(params.λ_u_sym)    * loss_dict['u_sym_term']**2)
    ℒ_ϕ_raw     = torch.mean(params.mask(params.λ_ϕ_sym)    * loss_dict['ϕ_sym_term']**2)
    ℒ_data_raw  = torch.mean(params.mask(params.λ_data)     * loss_dict['u_data_term']**2)

    # update adaptive weights every 100 epochs; reuse cached values otherwise
    if epoch % config.GRAD_NORM_EPOCH_INTERVAL == 0:
        # --- gradient norms of those scalars ---
        g_J     = gradientNormalize(ℒ_J_raw, PINN, params)
        g_Jy    = gradientNormalize(ℒ_Jy_raw, PINN, params)
        g_Σxy   = gradientNormalize(ℒ_Σxy_raw, PINN, params)
        g_Σyy   = gradientNormalize(ℒ_Σyy_raw, PINN, params)
        g_mass  = gradientNormalize(ℒ_mass_raw, PINN, params)
        g_BC    = gradientNormalize(ℒ_BC_raw, PINN, params)
        g_u     = gradientNormalize(ℒ_u_raw, PINN, params)
        g_ϕ     = gradientNormalize(ℒ_ϕ_raw, PINN, params)
        g_data  = gradientNormalize(ℒ_data_raw, PINN, params)

        gsum = g_J + g_Jy + g_Σxy + g_Σyy + g_mass + g_BC + g_u + g_ϕ + g_data

        Λ_J    = safeRatio(gsum, g_J)
        Λ_Jy   = safeRatio(gsum, g_Jy)
        Λ_Σxy  = safeRatio(gsum, g_Σxy)
        Λ_Σyy  = safeRatio(gsum, g_Σyy)
        Λ_mass = safeRatio(gsum, g_mass)
        Λ_BC   = safeRatio(gsum, g_BC)
        Λ_u    = safeRatio(gsum, g_u)
        Λ_ϕ    = safeRatio(gsum, g_ϕ)
        Λ_data = safeRatio(gsum, g_data)

        # EMA smoothing
        ξ = config.ξ
        Λ_J_new    = ξ*Λ_J_old    + (1-ξ)*Λ_J
        Λ_Jy_new   = ξ*Λ_Jy_old   + (1-ξ)*Λ_Jy
        Λ_Σxy_new  = ξ*Λ_Σxy_old  + (1-ξ)*Λ_Σxy
        Λ_Σyy_new  = ξ*Λ_Σyy_old  + (1-ξ)*Λ_Σyy
        Λ_mass_new = ξ*Λ_mass_old + (1-ξ)*Λ_mass
        Λ_BC_new   = ξ*Λ_BC_old   + (1-ξ)*Λ_BC
        Λ_u_new    = ξ*Λ_u_sym_old+ (1-ξ)*Λ_u
        Λ_ϕ_new    = ξ*Λ_ϕ_sym_old+ (1-ξ)*Λ_ϕ
        Λ_data_new = ξ*Λ_data_old + (1-ξ)*Λ_data

        # update globals
        Λ_J_old, Λ_Jy_old, Λ_Σxy_old, Λ_Σyy_old = Λ_J_new, Λ_Jy_new, Λ_Σxy_new, Λ_Σyy_new
        Λ_mass_old, Λ_BC_old, Λ_u_sym_old, Λ_ϕ_sym_old, Λ_data_old = Λ_mass_new, Λ_BC_new, Λ_u_new, Λ_ϕ_new, Λ_data_new

    # now apply base group weights + adaptive weights (cached or updated)
    ℒ_J    = (ℒ_J_raw    * config.Λ_PDEs) * Λ_J_old
    ℒ_Jy   = (ℒ_Jy_raw   * config.Λ_BCs)  * Λ_Jy_old
    ℒ_Σxy  = (ℒ_Σxy_raw  * config.Λ_PDEs) * Λ_Σxy_old
    ℒ_Σyy  = (ℒ_Σyy_raw  * config.Λ_PDEs) * Λ_Σyy_old
    ℒ_mass = (ℒ_mass_raw * config.Λ_BCs)  * Λ_mass_old
    ℒ_BC   = (ℒ_BC_raw   * config.Λ_BCs)  * Λ_BC_old
    ℒ_u    = (ℒ_u_raw    * config.Λ_BCs)  * Λ_u_sym_old
    ℒ_ϕ    = (ℒ_ϕ_raw    * config.Λ_BCs)  * Λ_ϕ_sym_old
    ℒ_data = (ℒ_data_raw * config.Λ_data) * Λ_data_old

    ℒ_total = ℒ_J + ℒ_Jy + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_BC + ℒ_u + ℒ_ϕ + ℒ_data
    ℒ_individuals = [ℒ_J, ℒ_Jy, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_BC, ℒ_u, ℒ_ϕ, ℒ_data]
    return ℒ_total, ℒ_individuals

# Gradient Norm Loss Scheme | Max Gradient Annealing
def lossGradientNormalizationMax(loss_dict, params, PINN, epoch):
    global Λ_J_old, Λ_Jy_old, Λ_Σxy_old, Λ_Σyy_old, Λ_mass_old, Λ_BC_old, Λ_u_sym_old, Λ_ϕ_sym_old, Λ_data_old
    ℒ_J_raw     = torch.mean(params.mask(params.λ_J)        * loss_dict['J_div_']**2)
    ℒ_Jy_raw    = torch.mean(params.mask(params.λ_Jy_wall)  * loss_dict['Jy_wall_']**2)
    ℒ_Σxy_raw   = torch.mean(params.mask(params.λ_Σxy)      * loss_dict['dΣxy_dy_']**2)
    ℒ_Σyy_raw   = torch.mean(params.mask(params.λ_Σyy)      * loss_dict['dΣyy_dy_']**2)
    ℒ_mass_raw  = torch.mean(params.mask(params.λ_mass)     * loss_dict['ϕ_bulk_term']**2)
    ℒ_BC_raw    = torch.mean(params.mask(params.λ_BC)       * loss_dict['ϕ_BC_term']**2)
    ℒ_u_raw     = torch.mean(params.mask(params.λ_u_sym)    * loss_dict['u_sym_term']**2)
    ℒ_ϕ_raw     = torch.mean(params.mask(params.λ_ϕ_sym)    * loss_dict['ϕ_sym_term']**2)
    ℒ_data_raw  = torch.mean(params.mask(params.λ_data)     * loss_dict['u_data_term']**2)

    # update adaptive weights every 100 epochs; reuse cached values otherwise
    if epoch % config.GRAD_NORM_EPOCH_INTERVAL == 0:
        # For reference residual (numerator): max abs
        g_J     = maxGradMagnitude(ℒ_J_raw, PINN, params)
        g_Σxy   = maxGradMagnitude(ℒ_Σxy_raw, PINN, params)
        g_Σyy   = maxGradMagnitude(ℒ_Σyy_raw, PINN, params)
        g_res_max = max(g_J, g_Σxy, g_Σyy)

        # For all other terms (denominator): mean abs
        g_Jy    = meanAbsGrad(ℒ_Jy_raw, PINN, params)
        g_mass  = meanAbsGrad(ℒ_mass_raw, PINN, params)
        g_BC    = meanAbsGrad(ℒ_BC_raw, PINN, params)
        g_u     = meanAbsGrad(ℒ_u_raw, PINN, params)
        g_ϕ     = meanAbsGrad(ℒ_ϕ_raw, PINN, params)
        g_data  = meanAbsGrad(ℒ_data_raw, PINN, params)

        Λ_J    = safeRatio(g_res_max, g_J)
        Λ_Jy   = safeRatio(g_res_max, g_Jy)
        Λ_Σxy  = safeRatio(g_res_max, g_Σxy)
        Λ_Σyy  = safeRatio(g_res_max, g_Σyy)
        Λ_mass = safeRatio(g_res_max, g_mass)
        Λ_BC   = safeRatio(g_res_max, g_BC)
        Λ_u    = safeRatio(g_res_max, g_u)
        Λ_ϕ    = safeRatio(g_res_max, g_ϕ)
        Λ_data = safeRatio(g_res_max, g_data)

        # EMA smoothing
        ξ = config.ξ
        Λ_J_new    = ξ*Λ_J_old    + (1-ξ)*Λ_J
        Λ_Jy_new   = ξ*Λ_Jy_old   + (1-ξ)*Λ_Jy
        Λ_Σxy_new  = ξ*Λ_Σxy_old  + (1-ξ)*Λ_Σxy
        Λ_Σyy_new  = ξ*Λ_Σyy_old  + (1-ξ)*Λ_Σyy
        Λ_mass_new = ξ*Λ_mass_old + (1-ξ)*Λ_mass
        Λ_BC_new   = ξ*Λ_BC_old   + (1-ξ)*Λ_BC
        Λ_u_new    = ξ*Λ_u_sym_old+ (1-ξ)*Λ_u
        Λ_ϕ_new    = ξ*Λ_ϕ_sym_old+ (1-ξ)*Λ_ϕ
        Λ_data_new = ξ*Λ_data_old + (1-ξ)*Λ_data

        # update globals
        Λ_J_old, Λ_Jy_old, Λ_Σxy_old, Λ_Σyy_old = Λ_J_new, Λ_Jy_new, Λ_Σxy_new, Λ_Σyy_new
        Λ_mass_old, Λ_BC_old, Λ_u_sym_old, Λ_ϕ_sym_old, Λ_data_old = Λ_mass_new, Λ_BC_new, Λ_u_new, Λ_ϕ_new, Λ_data_new

    # now apply base group weights + adaptive weights (cached or updated)
    ℒ_J    = (ℒ_J_raw    * config.Λ_PDEs) * Λ_J_old
    ℒ_Jy   = (ℒ_Jy_raw   * config.Λ_BCs)  * Λ_Jy_old
    ℒ_Σxy  = (ℒ_Σxy_raw  * config.Λ_PDEs) * Λ_Σxy_old
    ℒ_Σyy  = (ℒ_Σyy_raw  * config.Λ_PDEs) * Λ_Σyy_old
    ℒ_mass = (ℒ_mass_raw * config.Λ_BCs)  * Λ_mass_old
    ℒ_BC   = (ℒ_BC_raw   * config.Λ_BCs)  * Λ_BC_old
    ℒ_u    = (ℒ_u_raw    * config.Λ_BCs)  * Λ_u_sym_old
    ℒ_ϕ    = (ℒ_ϕ_raw    * config.Λ_BCs)  * Λ_ϕ_sym_old
    ℒ_data = (ℒ_data_raw * config.Λ_data) * Λ_data_old

    ℒ_total = ℒ_J + ℒ_Jy + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_BC + ℒ_u + ℒ_ϕ + ℒ_data
    ℒ_individuals = [ℒ_J, ℒ_Jy, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_BC, ℒ_u, ℒ_ϕ, ℒ_data]
    return ℒ_total, ℒ_individuals

# Basic Loss Scheme
def lossBaseline(loss_dict, params):
    # Individual losses
    ℒ_J                         = torch.mean(params.mask(params.λ_J)         * loss_dict['J_div_']**2)       * config.Λ_PDEs
    ℒ_Jy_wall                   = torch.mean(params.mask(params.λ_Jy_wall)   * loss_dict['Jy_wall_']**2)     * config.Λ_BCs
    ℒ_Σxy                       = torch.mean(params.mask(params.λ_Σxy)       * loss_dict['dΣxy_dy_']**2)     * config.Λ_PDEs
    ℒ_Σyy                       = torch.mean(params.mask(params.λ_Σyy)       * loss_dict['dΣyy_dy_']**2)     * config.Λ_PDEs
    ℒ_mass                      = torch.mean(params.mask(params.λ_mass)      * loss_dict['ϕ_bulk_term']**2)  * config.Λ_BCs
    ℒ_BC                        = torch.mean(params.mask(params.λ_BC)        * loss_dict['ϕ_BC_term']**2)    * config.Λ_BCs
    ℒ_u_sym                     = torch.mean(params.mask(params.λ_u_sym)     * loss_dict['u_sym_term']**2)   * config.Λ_BCs
    ℒ_ϕ_sym                     = torch.mean(params.mask(params.λ_ϕ_sym)     * loss_dict['ϕ_sym_term']**2)   * config.Λ_BCs
    ℒ_data                      = torch.mean(params.mask(params.λ_data)      * loss_dict['u_data_term']**2)  * config.Λ_data
    ℒ_ϕ_MSE                     = torch.mean(loss_dict['ϕ_MSE_term']**2)
    # NOTE: ℒ_mass does not need mean(...), as it is already scalars, but for the the sake of code consistiency, they are

    # Individual losses
    """ℒ_J                         = torch.mean(params.mask(params.λ_J)         * loss_dict['J_div_']**2)       * params.Λ_PDEs**2
    ℒ_Jy_wall                   = torch.mean(params.mask(params.λ_Jy_wall)   * loss_dict['Jy_wall_']**2)     * params.Λ_BCs**2
    ℒ_Σxy                       = torch.mean(params.mask(params.λ_Σxy)       * loss_dict['dΣxy_dy_']**2)     * params.Λ_PDEs**2
    ℒ_Σyy                       = torch.mean(params.mask(params.λ_Σyy)       * loss_dict['dΣyy_dy_']**2)     * params.Λ_PDEs**2
    ℒ_mass                      = torch.mean(params.mask(params.λ_mass)      * loss_dict['ϕ_bulk_term']**2)  * params.Λ_BCs**2
    ℒ_BC                        = torch.mean(params.mask(params.λ_BC)        * loss_dict['ϕ_BC_term']**2)    * params.Λ_BCs**2
    ℒ_u_sym                     = torch.mean(params.mask(params.λ_u_sym)     * loss_dict['u_sym_term']**2)   * params.Λ_BCs**2
    ℒ_ϕ_sym                     = torch.mean(params.mask(params.λ_ϕ_sym)     * loss_dict['ϕ_sym_term']**2)   * params.Λ_BCs**2
    ℒ_data                      = torch.mean(params.mask(params.λ_data)      * loss_dict['u_data_term']**2)  * params.Λ_data**2
    ℒ_ϕ_MSE                     = torch.mean(loss_dict['ϕ_MSE_term']**2)  * 1e5
    # NOTE: ℒ_mass does not need mean(...), as it is already scalars, but for the the sake of code consistiency, they are
    print("globals: ", params.Λ_PDEs.item(), params.Λ_BCs.item(), params.Λ_data.item(), ℒ_ϕ_MSE.item())"""

    # Compute loss
    ℒ = ℒ_J + ℒ_Jy_wall + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_BC + ℒ_u_sym + ℒ_ϕ_sym + ℒ_data
    ℒ_individuals = [ℒ_J, ℒ_Jy_wall, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_BC, ℒ_u_sym, ℒ_ϕ_sym, ℒ_data]
    # print("ℒ_ϕ_MSE: ", ℒ_ϕ_MSE.item())

    return ℒ, ℒ_individuals

# Normalized Losses Loss Scheme 
def lossNormalization(loss_dict, params):
    global ℒ_J_init, ℒ_Jy_init, ℒ_Σxy_init, ℒ_Σyy_init, ℒ_mass_init, ℒ_BC_init, ℒ_u_sym_init, ℒ_ϕ_sym_init, ℒ_data_init, ℒ_init

    # Individual losses
    ℒ_J                         = torch.mean(params.mask(params.λ_J)         * loss_dict['J_div_']**2)
    ℒ_Jy                        = torch.mean(params.mask(params.λ_Jy_wall)  * loss_dict['Jy_wall_']**2)
    ℒ_Σxy                       = torch.mean(params.mask(params.λ_Σxy)       * loss_dict['dΣxy_dy_']**2)
    ℒ_Σyy                       = torch.mean(params.mask(params.λ_Σyy)       * loss_dict['dΣyy_dy_']**2)
    ℒ_mass                      = torch.mean(params.mask(params.λ_mass)      * loss_dict['ϕ_bulk_term']**2)
    ℒ_BC                        = torch.mean(params.mask(params.λ_BC)        * loss_dict['ϕ_BC_term']**2)
    ℒ_u_sym                     = torch.mean(params.mask(params.λ_u_sym)     * loss_dict['u_sym_term']**2)
    ℒ_ϕ_sym                     = torch.mean(params.mask(params.λ_ϕ_sym)     * loss_dict['ϕ_sym_term']**2)
    ℒ_data                      = torch.mean(params.mask(params.λ_data)      * loss_dict['u_data_term']**2)
    # NOTE: ℒ_mass does not need mean(...), as it is already scalars, but for the the sake of code consistiency, they are
    
    # Initialize 
    if ℒ_J_init is None:        ℒ_J_init = ℒ_J.detach()
    if ℒ_Jy_init is None:       ℒ_Jy_init = ℒ_Jy.detach()
    if ℒ_Σxy_init is None:      ℒ_Σxy_init = ℒ_Σxy.detach()
    if ℒ_Σyy_init is None:      ℒ_Σyy_init = ℒ_Σyy.detach() 
    if ℒ_mass_init is None:     ℒ_mass_init = ℒ_mass.detach()
    if ℒ_BC_init is None:       ℒ_BC_init = ℒ_BC.detach()
    if ℒ_u_sym_init is None:    ℒ_u_sym_init = ℒ_u_sym.detach()
    if ℒ_ϕ_sym_init is None:    ℒ_ϕ_sym_init = ℒ_ϕ_sym.detach()
    if ℒ_data_init is None:     ℒ_data_init = ℒ_data.detach()
    if ℒ_init is None:          ℒ_init = ℒ_J_init + ℒ_Σxy_init + ℒ_Σyy_init + ℒ_mass_init + ℒ_BC_init + ℒ_u_sym_init + ℒ_ϕ_sym_init + ℒ_data_init

    # Normalize
    """ℒ_J                         = ℒ_J / ℒ_init
    ℒ_Σxy                       = ℒ_Σxy / ℒ_init
    ℒ_Σyy                       = ℒ_Σyy / ℒ_init
    ℒ_mass                      = ℒ_mass / ℒ_init
    ℒ_BC                        = ℒ_BC / ℒ_init
    ℒ_u_sym                     = ℒ_u_sym / ℒ_init
    ℒ_ϕ_sym                     = ℒ_ϕ_sym / ℒ_init
    ℒ_data                      = ℒ_data / ℒ_init"""

    # Normalize
    ℒ_J                         = ℒ_J / ℒ_J_init
    ℒ_Jy                        = ℒ_Jy / ℒ_Jy_init
    ℒ_Σxy                       = ℒ_Σxy / ℒ_Σxy_init
    ℒ_Σyy                       = ℒ_Σyy / ℒ_Σyy_init 
    ℒ_mass                      = ℒ_mass / ℒ_mass_init
    ℒ_BC                        = ℒ_BC / ℒ_BC_init
    ℒ_u_sym                     = ℒ_u_sym / ℒ_u_sym_init
    ℒ_ϕ_sym                     = ℒ_ϕ_sym / ℒ_ϕ_sym_init
    ℒ_data                      = ℒ_data / ℒ_data_init

    # Compute loss
    ℒ = ℒ_J + ℒ_Jy + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_BC + ℒ_u_sym + ℒ_ϕ_sym + ℒ_data
    ℒ_individuals = [ℒ_J, ℒ_Jy, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_BC, ℒ_u_sym, ℒ_ϕ_sym, ℒ_data]

    return ℒ, ℒ_individuals
