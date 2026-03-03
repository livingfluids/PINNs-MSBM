import torch.nn as nn
import loss_schemes

# NOTE: Problem Types
# Inverse Problem:      set Λ_data > 0
# Forward Problem:      set Λ_data = 0

# Processor
USE_GPU                 = False  # uses mps or cuda if available, fallback to cpu 

# Data Directory
DATA_DIR                = 'synthetic_data_example_1'  # directory for the problem case's data 

# Problem Case
CASE                    = 'learn cfl'  # 'learn cfl' or 'learn beta'

# Visualization
VISUALIZE_STEPS         = 100
SAVE_STEPS              = 500

# PINN Hyperparameters
EPOCHS: int             = 10000  # total epochs
NEURONS: int            = 64  # neurons 
BLOCKS: int             = 1  # PirateNet blocks (Wang et al.)
SCALE: float            = 10.0  # Fourier scale 
α_INIT: float           = 0.0  # learnable parameter | 0.0 for identity mapping, 1.0 for fully nonlinear mapping | 0.0 recommended (Wang et al.)
PINN_LR_INIT: float     = 1e-3  # PINN learning rate
COLL: int               = 500  # collocation points
ACTIVATION              = nn.Tanh()  # PINN activation function
GRAD_NORM_EPOCH_INTERVAL= 100

# Scheduler Hyperparameters
PATIENCE: int           = 25  # number of epochs with no improvement after which PINN learning rate will be reduced
FACTOR: float           = 0.9  # factor by which the PINN learning rate will be reduced
MIN_LR: float           = 1e-3  # lower bound on the PINN learning rate

# Loss Hyperparameters
λ_MASK                  = lambda λ: λ**2  # mask function for spatially self-sdaptive weights (McClenny & Braga-Neto)
λ_INIT                  = 1 # initial value spatially self-sdaptive weights
λ_LR_INIT: float        = 1  # spatially self-sdaptive weights learning rate (McClenny & Braga-Neto)
Λ_PDEs: float           = 1  # PDEs global weight
Λ_BCs: float            = 1  # BCs global weight
Λ_data: float           = 1  # data global weight
ξ: float                = 0.9 # moving average factor for global weighting updates | only for loss gradient normalization scheme

# Ray[Tune] Hyperparameter Space
REPORT_EVERY: int       = 150  # reports metric to Ray[Tune] every this many epochs
SAMPLE_RUNS: int        = 1  # number of samples to run together
PINN_LR_RANGE           = (1e-3, 1e-3)  # PINN learning rate
NEURONS_CHOICES: list   = [32, 64, 128]  # neurons
BLOCKS_CHOICES: list    = [1, 2, 3]  # blocks
COLL_CHOICES: list      = [100, 500, 1000]  # collocation points
ACTIV_CHOICES: list     = [nn.Tanh(), nn.ReLU(), nn.Sigmoid()]  # activation function
SCHEME_CHOICES: list    = ['baseline', 'grad_norm', 'loss_norm']  # loss scheme
OPTIM_CHOICES: list     = ['SOAP', 'Adam']  # PINN optimizer (does NOT apply to self-adaptive weights)
Λ_PDE_CHOICES: list     = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
Λ_BC_CHOICES: list      = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
Λ_DATA_CHOICES: list    = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
# Λ_PDE_RANGE             = (1, 1e5)
# Λ_BC_RANGE              = (1, 1e5)
# Λ_DATA_RANGE            = (1, 1e5)

# [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]