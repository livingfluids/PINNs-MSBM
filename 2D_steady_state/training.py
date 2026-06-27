import torch
import config
from adaptive_refinement import residualBasedAdaptiveRefinement
from soap import SOAP
from plot import History, plotResults
from loss import particlePartitionRatio, totalLoss
from mse import daughterProfileMSE, parentProfileMSE, profileFieldTitle, profileVisualizationField

# Train Model
def trainModel(trials, params, PINN, array, device):
    PINN_params     = list(PINN.parameters())
    PINN_optimizer  = SOAP(PINN_params)
    history         = History()
    profile_field   = profileVisualizationField()
    profile_title   = profileFieldTitle(profile_field)
    
    # Training Loop
    for epoch in range(config.EPOCHS):
        epoch += 1

        # 1. Zero-Grad
        PINN_optimizer.zero_grad()

        # 2. Backpropagate
        ℒ, L_ = totalLoss(trials=trials, params=params, PINN=PINN, array=array, device=device, epoch=epoch)
        ℒ.backward()
        
        torch.nn.utils.clip_grad_norm_(PINN_params, max_norm=1.0)

        # 3. Step
        PINN_optimizer.step()

        # Results 
        profile_MSE_out = daughterProfileMSE(trials, params, array, device, field=profile_field)
        if isinstance(profile_MSE_out, (tuple, list)): profile_MSE = profile_MSE_out[0]
        else: profile_MSE = profile_MSE_out
        profile_MSE_value = float(profile_MSE.detach().cpu().item())
        history.append_ϕ_mse(epoch=epoch, ϕ_mse=profile_MSE_value)
        # - 
        parent_profile_MSE = parentProfileMSE(trials, params, array, device, field=profile_field)
        parent_profile_MSE_value = float(parent_profile_MSE.detach().cpu().item())
        history.append_parent_ϕ_mse(epoch=epoch, ϕ_mse=parent_profile_MSE_value)
        # - 
        if (epoch % config.PLOT_EVERY == 0 or epoch == 1 or epoch == config.EPOCHS): plotResults(epoch=epoch, trials=trials, params=params, array=array, history=history, lift_force=L_)
        top, bottom = particlePartitionRatio(trials=trials, params=params, array=array, device=device)
        print(f'Epoch: {epoch}  Top Outlet ηᵖQ: {top:.3e}  Bottom Outlet ηᵖQ: {bottom:.3e}  Daughter {profile_title} MSE: {profile_MSE_value:.3e}  Parent {profile_title} MSE: {parent_profile_MSE_value:.3e}')

        # 4. Residual-Based Adaptive Refinement
        array, rar_stats = residualBasedAdaptiveRefinement(
            trials=trials,
            params=params,
            PINN=PINN,
            array=array,
            device=device,
            epoch=epoch,
        )
        if rar_stats is not None:
            print(
                "RAR "
                f"{rar_stats['method']} at epoch {epoch}: "
                f"added {rar_stats['added']} points "
                f"({rar_stats['interior_before']} -> {rar_stats['interior_after']} interior); "
                f"candidate residual mean/max {rar_stats['residual_mean']:.3e}/{rar_stats['residual_max']:.3e}; "
                f"selected mean/min {rar_stats['selected_residual_mean']:.3e}/{rar_stats['selected_residual_min']:.3e}"
            )
