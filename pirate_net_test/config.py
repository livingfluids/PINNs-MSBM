import torch

DEVICE = torch.device('cpu')
DTYPE = torch.float32
dp_dx_INIT = -2.5
NEURONS = 64 * 2
BLOCKS = 1
SCALE = 1.0
α_INIT = 1.0
PINN_LR_INIT = 1e-3
λ_LR_INIT = 1e-1
COLL = 500

PATIENCE = 100
FACTOR = 0.9
MIN_LR = 1e-3

EPOCHS = 10000

T_max = EPOCHS