import torch
from architecture import generate_trials, PINN
from params import generate_params
from training import train

# To-Do:
# add dp_dy to trainable parameters in train()

torch.manual_seed(0)

# Generate trial functions and parameters
params = generate_params()
trials = generate_trials(params=params)

print(PINN)

# Train model
train(trials=trials, params=params)
