import torch
import config
from loss import lossGradientNormalizationMax, ϕMSE
import plot
from soap import SOAP

# Define History Instance 
# history = visualize.History()

# Train Model
def trainModel(trials, params, PINN, device):
    def hasNonfiniteGrads(param_list):
        for p in param_list:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    # -
    PINN_params         = list(PINN.parameters())
    PINN_optimizer      = SOAP(PINN_params)
    # PINN_optimizer      = torch.optim.Adam(params=PINN_params, lr=config.PINN_LR_INIT)
    # λ_params            = [params.λ_J, params.λ_Jy_wall, params.λ_Σxy, params.λ_Σyy, params.λ_mass, params.λ_BC, params.λ_u_sym, params.λ_ϕ_sym, params.λ_data]
    # λ_optimizer         = torch.optim.Adam(params=λ_params, lr=config.λ_LR_INIT)

    # Loop 
    for epoch in range(config.EPOCHS):
        epoch += 1

        # 1. Zero-Grad
        PINN_optimizer.zero_grad()
        # λ_optimizer.zero_grad()

        # 2. Backpropagation
        ℒ, ℒ_individuals = lossGradientNormalizationMax(trials=trials, params=params, PINN=PINN, epoch=epoch, device=device)
        if not torch.isfinite(ℒ):
            if print: print(f"Non-finite loss detected at epoch {epoch}. Stopping.")
            break
        ℒ.backward()
        if hasNonfiniteGrads(PINN_params):
            if print: print(f"Non-finite gradients detected at epoch {epoch}. Stopping.")
            break

        # 3. Make gradient negative for λ parameters to achieve ascent, rather than descent
        # for λ in λ_params:
            # if λ.grad is not None: λ.grad = -λ.grad

        # Stabilize optimizer step
        pinn_clip = getattr(config, "PINN_GRAD_CLIP_NORM", 1.0)
        if pinn_clip is not None and pinn_clip > 0:
            torch.nn.utils.clip_grad_norm_(PINN_params, pinn_clip)

        # 4. Step
        PINN_optimizer.step()
        with torch.no_grad():
            β_min = getattr(config, "BETA_MIN", 0.25)
            β_max = getattr(config, "BETA_MAX", 3.0)
            params.β.clamp_(β_min, β_max)
        # λ_optimizer.step()

        # Update History
        # history.append(epoch, ℒ, ℒ_no_weights, ℒ_individuals, ℒ_individuals_no_weights)
        
        # Plot
        if epoch % 50 == 0: plot.plotResults(trials=trials, params=params)
        print(f'epoch: {epoch}  loss: {ℒ.item():.3f}')
        print("dp_dx_: ", params.dp_dx_.item(), "β: ", params.β.item())
        print("MSE: ", ϕMSE(trials, params).item())
