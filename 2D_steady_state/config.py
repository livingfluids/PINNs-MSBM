import torch
import torch.nn as nn

USE_GPU                 = True
# dtype used internally by geometry.py's wall-distance functions (pointToSegmentDistance /
# pointToSegmentNormalProjection). float64 avoids float32's ~1e-11 m rounding noise swamping
# the near-wall eps regularizer at these coordinate magnitudes; float32 skips the extra
# upcast/device-roundtrip cost (relevant on MPS, which has no native float64) but reintroduces
# that noise unless the eps in those functions is also raised well above ~1e-22.
WALL_DISTANCE_DTYPE      = torch.float32
NEURONS                 = 64
SCALE                   = 1
SCALES                  = [1, 2, 5]
ACTIVATION              = nn.Tanh()
N_PTS_PROPOSAL          = 30_000  # NOTE: approximately only 1/5 are actually used
N_PTS_BDR               = 50
CENTERLINE_SKEW_STRENGTH = 0.75
CENTERLINE_SKEW_RADIUS_FACTOR = 0.2  # influence width relative to the largest branch radius
INNER_WALL_SKEW_STRENGTH = 0.9
INNER_WALL_SKEW_RADIUS_FACTOR = 0.7  # influence radius relative to the largest branch radius
EPOCHS                  = 1_000_000
DATA_DIR                = 'MSBM_data'
Λ_PDEs: float           = 1  # PDEs global weight
Λ_BCs: float            = 1  # BCs global weight
Λ_data: float           = 1  # data global weight
Λ_PHI_NEUMANN: float    = 1  # wall ∂phi/∂n = 0 BC multiplier
GRAD_NORM_EPOCH_INTERVAL= 1000
PLOT_EVERY              = 100
SAVE_STEPS              = 100
PROFILE_VISUALIZATION_FIELD = "velocity"  # "phi", "velocity", or "lift"
DATA_LOSS_PROFILE_DISTANCE_FROM_V_CORNER = 0.0001
SHRINK_PARENT_TO_DATA_PROFILE = True
ξ = 0.9

# Residual-Based Adaptive Refinement (RAR)
RAR_ENABLED             = True
RAR_METHOD              = "rar-d"  # "rar-d" for distribution, "rar-g" for greedy top residuals
RAR_START_EPOCH         = 1000
RAR_EVERY               = 1000
RAR_ADD_POINTS          = 10
RAR_CANDIDATE_POINTS    = 20_000
RAR_BATCH_SIZE          = 1_024
RAR_MAX_INTERIOR_POINTS = 10_000  # 10_000
RAR_MIN_RESIDUAL        = 0.0
RAR_PROPOSAL_FACTOR     = 8

# RAR-D probability: p(x) proportional to residual(x)^k / mean(residual^k) + c
RAR_D_K                 = 2.0
RAR_D_C                 = 0.0

# For this multi-equation PINN, normalize each residual component over the candidate set
# before forming the refinement indicator so one equation scale does not dominate sampling.
RAR_NORMALIZE_COMPONENTS= True
RAR_WEIGHT_MIGRATION    = 1.0
RAR_WEIGHT_X_MOMENTUM   = 1.0
RAR_WEIGHT_Y_MOMENTUM   = 1.0
RAR_WEIGHT_CONTINUITY   = 1.0
RAR_WEIGHT_PHI_CONTINUITY = 0.0
