import torch
import config
import loss

EPS = 1e-12
PHI_FRAC_EPS = getattr(config, "PHI_FRAC_EPS", 1e-6)
ADAPTIVE_WEIGHT_MIN = getattr(config, "ADAPTIVE_WEIGHT_MIN", 1e-3)
ADAPTIVE_WEIGHT_MAX = getattr(config, "ADAPTIVE_WEIGHT_MAX", 1e3)

# Global variables for moving-average adaptive weights (same style as 1D)
Λ_J_old         = 1.0
Λ_Jwall_old     = 1.0
Λ_cont_old      = 1.0
Λ_Σx_old        = 1.0
Λ_Σy_old        = 1.0
Λ_noslip_old    = 1.0
Λ_inlet_grad_old= 1.0
Λ_ϕbulk_old     = 1.0
Λ_outlet_grad_old = 1.0
Λ_data_old      = 1.0   # if you later add data loss

def lossGradientNormalizationMax(trials, params, PINN, array, epoch, device):
    global Λ_J_old, Λ_Jwall_old, Λ_Σx_old, Λ_Σy_old, Λ_cont_old, Λ_inlet_grad_old, Λ_noslip_old, Λ_ϕbulk_old

    # Pull all residual terms from one shared forward/graph build in loss.py
    J_div_, J_wall_, Σx_div_, Σy_div_, continuity, noslip_loss, inlet_grad_loss, ϕ_bulk_loss = loss.computeLossTerms(
        trials=trials,
        params=params,
        array=array,
        device=device,
    )

    def _mean_sq(t):
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, device=device, dtype=torch.float32)
        if t.numel() == 0:
            return torch.zeros((), device=device)
        return torch.mean(t**2)

    ℒ_J_raw = _mean_sq(J_div_)
    ℒ_Jwall_raw = _mean_sq(J_wall_)
    ℒ_cont_raw = _mean_sq(continuity)
    ℒ_Σx_raw = _mean_sq(Σx_div_)
    ℒ_Σy_raw = _mean_sq(Σy_div_)
    ℒ_noslip_raw = _mean_sq(noslip_loss)
    ℒ_inlet_raw = _mean_sq(inlet_grad_loss)
    ℒ_ϕbulk_raw = _mean_sq(ϕ_bulk_loss)

    if epoch % config.GRAD_NORM_EPOCH_INTERVAL == 0:
        g_J = maxGradMagnitude(ℒ_J_raw, PINN, params)
        g_Σx = maxGradMagnitude(ℒ_Σx_raw, PINN, params)
        g_Σy = maxGradMagnitude(ℒ_Σy_raw, PINN, params)
        g_res_max = torch.max(torch.stack([g_J, g_Σx, g_Σy]))

        g_Jwall = meanAbsGrad(ℒ_Jwall_raw, PINN, params)
        g_cont = meanAbsGrad(ℒ_cont_raw, PINN, params)
        g_noslip = meanAbsGrad(ℒ_noslip_raw, PINN, params)
        g_inlet_grad = meanAbsGrad(ℒ_inlet_raw, PINN, params)
        g_ϕbulk = meanAbsGrad(ℒ_ϕbulk_raw, PINN, params)

        Λ_J = clampAdaptiveWeight(safeRatio(g_res_max, g_J))
        Λ_Jwall = clampAdaptiveWeight(safeRatio(g_res_max, g_Jwall))
        Λ_cont = clampAdaptiveWeight(safeRatio(g_res_max, g_cont))
        Λ_Σx = clampAdaptiveWeight(safeRatio(g_res_max, g_Σx))
        Λ_Σy = clampAdaptiveWeight(safeRatio(g_res_max, g_Σy))
        Λ_noslip = clampAdaptiveWeight(safeRatio(g_res_max, g_noslip))
        Λ_inlet_grad = clampAdaptiveWeight(safeRatio(g_res_max, g_inlet_grad))
        Λ_ϕbulk = clampAdaptiveWeight(safeRatio(g_res_max, g_ϕbulk))

        ξ = float(getattr(config, "ξ", 0.9))
        Λ_J_old = ξ * Λ_J_old + (1 - ξ) * Λ_J.item()
        Λ_Jwall_old = ξ * Λ_Jwall_old + (1 - ξ) * Λ_Jwall.item()
        Λ_cont_old = ξ * Λ_cont_old + (1 - ξ) * Λ_cont.item()
        Λ_Σx_old = ξ * Λ_Σx_old + (1 - ξ) * Λ_Σx.item()
        Λ_Σy_old = ξ * Λ_Σy_old + (1 - ξ) * Λ_Σy.item()
        Λ_noslip_old = ξ * Λ_noslip_old + (1 - ξ) * Λ_noslip.item()
        Λ_inlet_grad_old = ξ * Λ_inlet_grad_old + (1 - ξ) * Λ_inlet_grad.item()
        Λ_ϕbulk_old = ξ * Λ_ϕbulk_old + (1 - ξ) * Λ_ϕbulk.item()

    ℒ_J = (ℒ_J_raw * config.Λ_PDEs) * Λ_J_old
    ℒ_Jwall = (ℒ_Jwall_raw * config.Λ_BCs) * Λ_Jwall_old
    ℒ_cont = (ℒ_cont_raw * config.Λ_PDEs) * Λ_cont_old
    ℒ_Σx = (ℒ_Σx_raw * config.Λ_PDEs) * Λ_Σx_old
    ℒ_Σy = (ℒ_Σy_raw * config.Λ_PDEs) * Λ_Σy_old
    ℒ_noslip = (ℒ_noslip_raw * config.Λ_BCs) * Λ_noslip_old
    ℒ_inlet_grad = (ℒ_inlet_raw * config.Λ_BCs) * Λ_inlet_grad_old
    ℒ_ϕbulk = (ℒ_ϕbulk_raw * config.Λ_BCs) * Λ_ϕbulk_old

    ℒ = ℒ_J * 0 + ℒ_Jwall * 0 + ℒ_cont + ℒ_Σx + ℒ_Σy + ℒ_noslip * 10 + ℒ_inlet_grad + ℒ_ϕbulk * 0 
    ℒ_no_weights = ℒ_J_raw + ℒ_Jwall_raw + ℒ_cont_raw + ℒ_Σx_raw + ℒ_Σy_raw + ℒ_noslip_raw + ℒ_inlet_raw + ℒ_ϕbulk_raw

    ℒ_individuals = [
        ℒ_J,
        ℒ_Jwall,
        ℒ_cont,
        ℒ_Σx,
        ℒ_Σy,
        ℒ_noslip,
        ℒ_inlet_grad,
        ℒ_ϕbulk,
    ]
    ℒ_individuals_no_weights = [
        ℒ_J_raw,
        ℒ_Jwall_raw,
        ℒ_cont_raw,
        ℒ_Σx_raw,
        ℒ_Σy_raw,
        ℒ_noslip_raw,
        ℒ_inlet_raw,
        ℒ_ϕbulk_raw,
    ]
    return ℒ, ℒ_no_weights, ℒ_individuals, ℒ_individuals_no_weights

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
