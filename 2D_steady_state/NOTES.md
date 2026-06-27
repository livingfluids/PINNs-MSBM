
SBM converges well, though with small wall artifacts
MSBM converges well elsewhere in the bifurcation except at the inner walls

OpenFOAM Lift Force: lift direction is defined from the gradient of a wall-distance field.
PINN Life Force: lift direction is defined from the normal of the nearest segment, with fanning for segment endpoints

Various changes to the PINN codebase do not seem to change the general convergence:
- removed sigmoid constraint on phi trial function
- mask lift force at V corner within a small radius
- toggled on and off the phi neumann and phi dot n loss terms
- extended fourier scaling from 1 to simultaneous [1, 5, 10] scaling

- check velocity field profiles 


CFL and bulk phi known, shear stress unknown
https://arxiv.org/abs/2606.06313 
















--- Epoch 1100 (Model Converged To Usual Solution)---
RAW LOSSES:
  migration:   3.1096e-05
  x_momentum:  7.0098e-02
  y_momentum:  1.0027e-02
  Jn_wall:     1.8177e-09
  no_slip:     3.4745e-04
  u_data:      2.7159e-05
  phi_data:    1.0771e-04
GRAD-NORM WEIGHTS (Λ):
  Λ_migr:   2.2593e+05
  Λ_xmom:   4.6444e+01
  Λ_ymom:   2.0262e+02
  Λ_Jn:     7.6995e+08
  Λ_noslip: 3.7934e+04
  Λ_udata:  9.2602e+03
  Λ_phidata:6.8580e+04
EFFECTIVE WEIGHTED LOSSES:
  migration:   7.0257e+00
  x_momentum:  3.2556e+00
  y_momentum:  2.0317e+00
  Jn_wall:     1.3995e+00
  no_slip:     1.3180e+01
  u_data:      2.5150e-01
  phi_data:    7.3869e+00
GRAD MAGNITUDES:
  g_mi_max:  4.8183e-04
  g_xm_max:  1.2312e+00
  g_ym_max:  7.0977e-01
  g_res_max: 1.2312e+00
  g_no:      3.4767e-05
  g_ud:      1.1324e-04
  g_pd:      6.4017e-06
PHYSICAL DIAGNOSTICS (global):
  ϕ mean:      0.1941
  ϕ max:       0.4798
  ϕ min:       0.0858
  |∇·J| mean:  1.2174e-03
  |ϕU + J flux x| mean: 7.5632e-02
  |ϕU + J flux y| mean: 2.8454e-02
  migration residual mean: 3.4570e-03
  migration residual max:  4.3918e-02
BIFURCATION-LOCAL (within 0.5*R0 of V-corner, N=123):
  ϕ mean (near):        0.3193
  ϕ max (near):         0.4679
  migration mean (near):1.3471e-02
  migration max (near): 4.3918e-02
  |J| mean (near):      1.2780e-03
  ηN mean (near):       5.2192e-01
  f(ϕ) mean (near):     1.9174e-01
AWAY FROM BIFURCATION (N=2472):
  ϕ mean (far):         0.1879
  ϕ max (far):          0.4798
  migration mean (far): 2.9587e-03

-Experiments & Changes Attempted- 

Physics & Equation Formulation

Switched migration equation from non-conservative ∇·J + U·∇ϕ = 0 to fully conservative ∇·(ϕU + J) = 0 — improved but didn't resolve peaks
Made dΣpyx_dx_ explicit rather than reusing dΣpxy_dx_ to remove implicit symmetry assumption
Excluded lift force from migration flux (currently testing without it)
Tested zero Reynolds number
Added ϕ·∇·U penalty term for high-ϕ continuity violations
Switched to streamfunction-pressure representation — major structural improvement, guaranteed divergence-free velocity field, most impactful change made

Loss Scheme

Tested maxGradMagnitude vs meanAbsGrad normalization schemes
Included/excluded various loss terms: flux conservation, flow partition ratio, bulk conservation
Added profile MSEs (daughter and parent) directly into the total loss temporarily for diagnostics
Added V-corner no-slip exclusion radius for both Jn and no-slip terms
Discovered and corrected inverted EMA in grad-norm weight update — was slow-adapting (wrong direction per paper), corrected to fast-adapting per Wang et al. 2020
Added g_no_max to the g_res_max reference
Attempted Λ cap to prevent no-slip from dominating
Increased no-slip weight 10x, 100x — caused velocity field collapse toward zero
Added corner-proximity weighting to migration loss

Geometry & Collocation

Fixed hardcoded V-corner skew target (0.00015, 0.0) to use dynamically computed V-corner location
Increased/varied V_CORNER_SKEW_STRENGTH and skew radius
Increased N_PTS_BDR and N_PTS_PROPOSAL
Added centerline skew
Confirmed 18 collocation points within 0.1*R0 of V-corner — not zero but sparse
Confirmed domain mask does not have a gap at the V-corner junction

Architecture

Tested 3 vs 5 hidden layers
Tested various NEURONS (32, 48, 64) and SCALE (1, 3, 5) configurations
Attempted distance-weighted hard no-slip trial function (parabolic factor) — caused SOAP ill-conditioning
Attempted tanh wall distance factor — shape errors and memory issues
Attempted input augmentation with wall distance channel — dimension mismatch errors resolved but memory issues
Current: standard Fourier features with SCALE=1, suspect this may be too low to represent sharp peaks

Optimizer

SOAP with various lr (1e-3, 3e-3, 5e-4), betas ((0.95,0.95) vs (0.9,0.999)), weight_decay (0.01, 1e-3, 0.0)
Added progressive jitter ladder to SOAP get_orthogonal_matrix for ill-conditioning robustness
Added SVD fallback to SOAP preconditioner
Reduced GRAD_NORM_EPOCH_INTERVAL from 100 to various values

Diagnostics Run

Printed raw losses, Λ weights, effective losses, grad magnitudes at convergence
Key finding: migration raw loss tiny but Λ_migr enormous (~1e6), indicating weak gradient leverage
Key finding: no-slip grad magnitude ~35,000x smaller than momentum — wall BC not enforcing effectively in streamfunction formulation
Key finding: migration residual 4.6x larger near V-corner than away from it — PDE not satisfied locally at bifurcation
Key finding: ϕ max exists at correct magnitude (~0.47) near corner but spatially displaced from inner wall
Key finding: γ̇ underpredicted at walls — upstream cause of incorrect migration driving force
Continuity raw loss significant at convergence before streamfunction fix

Pending / Next Steps

Test higher Fourier scale (SCALE = 5 or 10) or multi-scale Fourier features
Reformulate no-slip as direct Dirichlet condition on ψ rather than on derived u, v
Obtain OpenFOAM case files from collaborator to verify constitutive model, BCs, and nondimensionalization match exactly
Verify u_max nondimensionalization is not underestimating true domain velocity maximum