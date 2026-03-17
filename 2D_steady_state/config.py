import numpy as np
import torch.nn as nn

USE_GPU                 = True
NEURONS                 = 96
SCALE                   = 1
ACTIVATION              = nn.Tanh()
N_PTS                   = 15_000
N_PTS_BDR               = 50
EPOCHS                  = 10_000
DATA_DIR                = 'fakedata1'
Λ_PDEs: float           = 1  # PDEs global weight
Λ_BCs: float            = 1  # BCs global weight
Λ_data: float           = 1  # data global weight
GRAD_NORM_EPOCH_INTERVAL= 100

Λ_PDEs: float           = 1  # PDEs global weight
Λ_BCs: float            = 1  # BCs global weight
Λ_data: float           = 1  # data global weight

ξ = 0.9