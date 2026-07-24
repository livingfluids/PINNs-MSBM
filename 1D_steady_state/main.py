import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
torch.manual_seed(0)
from architecture import buildTrials, buildSimplePINN
from params import buildParams
from training import trainModel
import config

# Processor 
if config.USE_GPU:
    if torch.backends.mps.is_available():   device = torch.device("mps")
    elif torch.cuda.is_available():         device = torch.device("cuda")
    else:                                   device = torch.device("cpu")
else:                                       device = torch.device("cpu")
print(f"Using device: {device}")

# Build Model
PINN    = buildSimplePINN(neurons=config.NEURONS).to(device)
params  = buildParams(device=device)
trials  = buildTrials(params=params, device=device, PINN=PINN)

# Train Model
trainModel(trials=trials, params=params, PINN=PINN, device=device)
