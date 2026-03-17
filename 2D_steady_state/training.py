import torch
import config
from soap import SOAP
from loss_scheme import lossGradientNormalizationMax
from plot import History, plotResults

# Train Model
def trainModel(trials, params, PINN, array, device):
    PINN_params = list(PINN.parameters())
    PINN_optimizer = SOAP(PINN_params)
    # SOAP(PINN_params)
    plot_every = int(getattr(config, "PLOT_EVERY", 50))
    # - 
    for epoch in range(config.EPOCHS):
        epoch += 1

        # 1. Zero-Grad
        PINN_optimizer.zero_grad()

        # 2. Backpropagate
        ℒ = lossGradientNormalizationMax(trials=trials, params=params, PINN=PINN, array=array, device=device, epoch=epoch)
        ℒ.backward()

        # 3. Step
        PINN_optimizer.step()

        # Results 
        if plot_every > 0 and (epoch % plot_every == 0 or epoch == 1 or epoch == config.EPOCHS): plotResults(epoch=epoch, trials=trials, params=params, array=array)
        print(epoch)
