import torch
from pathlib import Path
import config
from loss import lossGradientNormalizationMax, ϕMSE
from soap import SOAP
import plot
import paths

# Define History Instance 
history = plot.History()


def saveModel(epoch, PINN, params, history):
    results_dir = paths.resu_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "case": config.CASE,
        "model_state_dict": PINN.state_dict(),
        "params": {
            name: value.detach().cpu().clone()
            for name, value in vars(params).items()
            if torch.is_tensor(value)
        },
        "history": {
            "epochs": history.epochs,
            "total": history.total,
            "total_no_weight": history.total_no_weight,
            "individuals": history.individuals,
            "individuals_no_weight": history.individuals_no_weight,
            "βs": history.βs,
            "MSEs": history.MSEs,
        },
    }

    save_path = results_dir / f"{config.DATA_DIR}_{config.CASE}_{epoch}_inverse.pt"
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint: {save_path}")

# Train Model
def trainModel(trials, params, PINN, device):
    if config.CASE == 'learn_beta': PINN_params = list(PINN.parameters()) + [params.β]
    elif config.CASE == 'learn_cfl': PINN_params = list(PINN.parameters())
    else: raise ValueError(f"Invalid CASE: {config.CASE}")
    PINN_optimizer  = SOAP(PINN_params)
    # - 
    λ_params = [params.λ_J, params.λ_Jy_wall, params.λ_Σxy, params.λ_Σyy, params.λ_mass, params.λ_u_sym, params.λ_ϕ_sym, params.λ_data]
    if config.CASE == 'learn_beta' or config.PROBLEM == 'forward': λ_params.insert(5, params.λ_BC)
    λ_optimizer = torch.optim.Adam(params=λ_params, lr=config.λ_LR_INIT)

    # Training Loop 
    for epoch in range(config.EPOCHS):
        epoch += 1

        # 1. Zero-Grad
        PINN_optimizer.zero_grad()
        if epoch % config.SA_EPOCH == 0: λ_optimizer.zero_grad()

        # 2. Backpropagate
        ℒ, ℒ_no_weights, ℒ_individuals, ℒ_individuals_no_weights = lossGradientNormalizationMax(trials=trials, params=params, PINN=PINN, device=device, epoch=epoch)
        ℒ.backward()

        # 3. Make gradient negative for λ parameters to achieve ascent, rather than descent
        if epoch % config.SA_EPOCH == 0: 
            for λ in λ_params:
                if λ.grad is not None: λ.grad = -λ.grad

        # 4. Step
        PINN_optimizer.step()
        if epoch % config.SA_EPOCH == 0: λ_optimizer.step()

        # Update History
        history.append(epoch, ℒ, ℒ_no_weights, ℒ_individuals, ℒ_individuals_no_weights, params.β, ϕMSE(trials, params))
        
        # Plot
        #if epoch % config.VISUALIZE_STEPS == 0: plot.plotResults(epoch=epoch, trials=trials, params=params, history=history)
        if epoch % 1000 == 0: saveModel(epoch=epoch, PINN=PINN, params=params, history=history)
        print(f'epoch: {epoch}  loss: {ℒ_no_weights.item():.3f}')
        print("dp_dx_: ", params.dp_dx_.item(), "β: ", params.β.item(), "MSE: ", ϕMSE(trials, params).item())
