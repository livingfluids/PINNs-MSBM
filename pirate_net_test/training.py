import torch
from architecture import PINN
import config
from loss import total_loss
from visualize import plot

# try other scheduler types

def train(trials, params): 
    # Initialize
    PINN_params         = list(PINN.parameters()) + [params.dp_dx]
    PINN_optimizer      = torch.optim.Adam(params=PINN_params, lr=config.PINN_LR_INIT)
    # PINN_scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(PINN_optimizer, factor=config.FACTOR, patience=config.PATIENCE, min_lr=config.MIN_LR)
    PINN_scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(PINN_optimizer, T_max=config.T_max)
    λ_params            = [params.λ_J, params.λ_Σxy, params.λ_Σyy, params.λ_mass, params.λ_symmetry, params.λ_data]
    λ_optimizer         = torch.optim.Adam(params=λ_params, lr=config.λ_LR_INIT)
    λ_scheduler         = torch.optim.lr_scheduler.CosineAnnealingLR(λ_optimizer, T_max=config.T_max)

    # Loop 
    for epoch in range(config.EPOCHS):
        epoch += 1

        PINN_optimizer.zero_grad()
        λ_optimizer.zero_grad()
        ℒ, ℒ_no_weights, ℒ_individual = total_loss(trials=trials, params=params)
        ℒ.backward()

        # Make gradient negative for λ parameters to achieve ascent, rather than descent
        for λ in λ_params:
            if λ.grad is not None:
                λ.grad = -λ.grad

        PINN_optimizer.step()
        PINN_scheduler.step(ℒ_no_weights.item())
        λ_optimizer.step()
        # λ_scheduler.step()
        
        plot(epoch=epoch, trials=trials, params=params)
        print(f'epoch: {epoch}  loss: {ℒ_no_weights.item():.3f}  PINN lr: {PINN_scheduler.get_last_lr()[0]:.3e}  λ lr: {λ_scheduler.get_last_lr()[0]:.3e}')
        print('ℒ_individual: ', {l.item() for l in ℒ_individual})
        print('dp_dx: ', params.dp_dx)