import torch
import config
import loss

# Script Constants 
EPS = 1e-12

# Initialize
Λ_migr_old      = 1.0
Λ_xmom_old      = 1.0
Λ_ymom_old      = 1.0
Λ_cont_old      = 1.0
Λ_noslip_old    = 1.0
Λ_inlet_old     = 1.0
Λ_ϕbulk_old     = 1.0
Λ_udata_old     = 1.0
Λ_cont2_old     = 1.0

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

# Max Loss Gradient Normalization
def lossGradientNormalizationMax(trials, params, PINN, array, epoch, device):
    global Λ_migr_old, Λ_xmom_old, Λ_ymom_old, Λ_cont_old, Λ_noslip_old, Λ_inlet_old, Λ_ϕbulk_old, Λ_udata_old, Λ_cont2_old

    migration, x_momentum, y_momentum, continuity = loss.PDELoss(trials, params, array, device)
    no_slip = loss.noSlipBCLoss(trials, params, array, device)
    ϕ_bulk = loss.ϕBulkLoss(trials, params, array, device)
    u_data = loss.uDataLoss(trials, params, array, device)
    cont_2 = loss.contLoss(trials, params, array, device)

    ℒ_migration_raw = torch.mean(migration**2)
    ℒ_x_momentum_raw = torch.mean(x_momentum**2)
    ℒ_y_momentum_raw = torch.mean(y_momentum**2)
    ℒ_continuity_raw = torch.mean(continuity**2)
    ℒ_no_slip_raw = torch.mean(no_slip**2)
    ℒ_ϕ_bulk_raw = torch.mean(ϕ_bulk**2)
    ℒ_u_data_raw = torch.mean(u_data**2)
    ℒ_cont_2_raw = torch.mean(cont_2**2)

    if epoch % config.GRAD_NORM_EPOCH_INTERVAL == 0:
        g_mi = maxGradMagnitude(ℒ_migration_raw, PINN)
        g_xm = maxGradMagnitude(ℒ_x_momentum_raw, PINN)
        g_ym = maxGradMagnitude(ℒ_y_momentum_raw, PINN)
        g_co = maxGradMagnitude(ℒ_continuity_raw, PINN)
        g_res_max = torch.max(torch.stack([g_mi, g_xm, g_ym, g_co]))
        # - 
        g_no = meanAbsGrad(ℒ_no_slip_raw, PINN)
        g_ϕb = meanAbsGrad(ℒ_ϕ_bulk_raw, PINN)
        g_ud = meanAbsGrad(ℒ_u_data_raw, PINN)
        g_c2 = meanAbsGrad(ℒ_cont_2_raw, PINN)
        # - 
        Λ_mi = g_res_max / (g_mi + EPS)
        Λ_xm = g_res_max / (g_xm + EPS)
        Λ_ym = g_res_max / (g_ym + EPS)
        Λ_co = g_res_max / (g_co + EPS)
        Λ_no = g_res_max / (g_no + EPS)
        Λ_ϕb = g_res_max / (g_ϕb + EPS)
        Λ_ud = g_res_max / (g_ud + EPS)
        Λ_c2 = g_res_max / (g_c2 + EPS)
        # - 
        ξ = config.ξ
        Λ_migr_old = ξ * Λ_migr_old + (1 - ξ) * Λ_mi.item()
        Λ_xmom_old = ξ * Λ_xmom_old + (1 - ξ) * Λ_xm.item()
        Λ_ymom_old = ξ * Λ_ymom_old + (1 - ξ) * Λ_ym.item()
        Λ_cont_old = ξ * Λ_cont_old + (1 - ξ) * Λ_co.item()
        Λ_noslip_old = ξ * Λ_noslip_old + (1 - ξ) * Λ_no.item()
        Λ_ϕbulk_old = ξ * Λ_ϕbulk_old + (1 - ξ) * Λ_ϕb.item()
        Λ_udata_old = ξ * Λ_udata_old + (1 - ξ) * Λ_ud.item()
        Λ_cont2_old = ξ * Λ_cont2_old + (1 - ξ) * Λ_c2.item()

    ℒ_mi = (ℒ_migration_raw * config.Λ_PDEs) * Λ_migr_old
    ℒ_xm = (ℒ_x_momentum_raw * config.Λ_PDEs) * Λ_xmom_old
    ℒ_ym = (ℒ_y_momentum_raw * config.Λ_PDEs) * Λ_ymom_old
    ℒ_co = (ℒ_continuity_raw * config.Λ_PDEs) * Λ_cont_old
    ℒ_no = (ℒ_no_slip_raw * config.Λ_BCs) * Λ_noslip_old
    ℒ_ϕb = (ℒ_ϕ_bulk_raw * config.Λ_BCs) * Λ_ϕbulk_old
    ℒ_ud = (ℒ_u_data_raw * config.Λ_data) * Λ_udata_old
    ℒ_c2 = (ℒ_cont_2_raw * config.Λ_BCs) * Λ_cont2_old

    ℒ = ℒ_mi + ℒ_xm + ℒ_ym + ℒ_co * 100 + ℒ_no + ℒ_ϕb + ℒ_ud + ℒ_c2

    return ℒ
