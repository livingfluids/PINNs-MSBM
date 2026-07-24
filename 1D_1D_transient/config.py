# ╭────────────────────────────────────────────╮
# | Configure hyperparameters from this script |
# ╰────────────────────────────────────────────╯

import torch.nn as nn
import torch

# NOTE: Problem Types
# Inverse Problem:      set Λ_data > 0
# Forward Problem:      set Λ_data = 0

# Processor
USE_GPU                 = True  # uses mps or cuda if available, fallback to cpu 

# Data Directory
DATA_DIR                = 'transientdata'  # directory for the problem case's data 

# PINN Hyperparameters
EPOCHS: int             = 10000  # total epochs
NEURONS: int            = 96  # neurons 
SCALE: float            = 1.0  # Fourier scale 
PINN_LR_INIT: float     = 1e-3  # PINN learning rate
t_COLL: int             = 100  # collocation points
y_COLL: int             = 100  # collocation points
ACTIVATION              = nn.Tanh()  # PINN activation function
T_COLL_EXPONENT: float  = 1.0  # t collocation warp; <1 skews toward higher t, >1 skews toward lower t

GRAD_NORM_EPOCH_INTERVAL = 100

# Scheduler Hyperparameters
PATIENCE: int           = 25  # number of epochs with no improvement after which PINN learning rate will be reduced
FACTOR: float           = 0.9  # factor by which the PINN learning rate will be reduced
MIN_LR: float           = 1e-3  # lower bound on the PINN learning rate

# Loss Hyperparameters
λ_MASK                  = lambda λ: λ**2  # mask function for spatially self-sdaptive weightsv(McClenny & Braga-Neto)
λ_INIT: float           = 1
λ_LR_INIT: float        = 1e-3  # spatially self-sdaptive weights learning rate (McClenny & Braga-Neto)
Λ_PDEs: float           = 1  # PDEs global weight
Λ_BCs: float            = 1  # BCs global weight
Λ_data: float           = 1  # data global weight
ξ: float                = 0.9 # moving average factor for global weighting updates | only for loss gradient normalization scheme

