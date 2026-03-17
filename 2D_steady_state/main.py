import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
torch.manual_seed(0)
from architecture import buildTrials, buildSimplePINN
from params import buildParams
from geometry import buildInteriorArray, buildBifurcationArray
from training import trainModel
from visualize import visualizeArray
import config
import numpy as np

"""
Loss:
∇⋅U = 0             Incompressibility
(U⋅∇)U - ∇⋅Σ = 0    Momentum 
∇⋅J = 0             Suspension 

To-Do:
normalize properly
outlet term of some sort (velocity, ensure continuity rule from physics 1 is obeyed, AV=AV)
introduce phi terms
plot/track loss
"""

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
array   = buildBifurcationArray(params=params, device=device)
trials  = buildTrials(params=params, device=device, PINN=PINN)
print(params.R0)

# Visualize Array
visualizeArray(array)

# Train Model
trainModel(trials=trials, params=params, PINN=PINN, array=array, device=device)