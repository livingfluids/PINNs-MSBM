import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
torch.manual_seed(0)
from architecture import buildTrials, buildSimplePINN
from params import buildParams
from geometry import buildBifurcationArray
from loss import innerBifurcationVCorner
from training import trainModel
from visualize import visualizeArray
import config

def main():
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

    v_corner = innerBifurcationVCorner(params, device, dtype=torch.float32)
    dists = torch.linalg.norm(array["interior_array"] - v_corner.unsqueeze(0), dim=1)
    print(f"Points within 0.1*R0 of V-corner: {(dists < 0.1*float(params.R0)).sum()}")

    # Visualize Array
    visualizeArray(array)

    # Train Model
    trainModel(trials=trials, params=params, PINN=PINN, array=array, device=device)

if __name__ == "__main__": main()
