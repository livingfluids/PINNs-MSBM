import torch
import config

# Placeholders
# formalize torch.size... comments
# formalize all comments 

# NOTE: Normalized values contain '*_'

def grad(y, f):
    y = y if y.ndim == 2 else y.unsqueeze(1)
    """
    f = f if f.ndim == 2 else f.unsqueeze(1)

    dy = y[1:] - y[:-1]
    df = f[1:] - f[:-1]
    dfdystar_center = (f[2:] - f[:-2]) / (y[2:] - y[:-2])
    dfdystar_left = df[0:1] / dy[0:1]
    dfdystar_right = df[-1:] / dy[-1:]
    df_dy = torch.cat([dfdystar_left, dfdystar_center, dfdystar_right], dim=0)  # torch.Size([y, 1]) 
    """
    df_dy = torch.autograd.grad(f, y, torch.ones_like(f), create_graph=True)[0]  # torch.Size([y, 1])
    return df_dy

def u_data_loss(trials, params): return trials.u_trial(params.y_data_) - params.u_data_

def physics_loss(trials, params):  # physics ensures ∇⋅J = ∇⋅Σ = 0
    # Initialize 
    y_ = params.y_coll_        # already normalized 
    u_ = trials.u_trial(y_)     # already normalized 
    ϕ_ = trials.ϕ_trial(y_)    # ϕ itself is not normalized, but we do calculate it here along y_
    A = 2 * params.a / params.H
    p_ = params.p * params.H / (2 * params.η0 * params.u_max)
    zero = torch.zeros_like(y_, device=config.DEVICE)  # torch.Size([y, 1])

    # Normal stress viscosity (ηₙ(ϕ))
    def ηN(ϕ): return params.Kn * (ϕ/params.ϕ_max)**2 * (1 - ϕ/params.ϕ_max)**(-2)  # torch.Size([y_.shape[0], 1]), a scalar for each y

    # Shear viscosity of the particle phase (ηₚ(ϕ))
    def ηp(ϕ):
        ηs = (1 - ϕ/params.ϕ_max)**(-2)
        return ηs - 1  # torch.Size([y, 1]), a scalar for each y

    # Sedimentation hinderence function for mobility of particle phase (f(ϕ))
    def f(ϕ): return (1 - ϕ/params.ϕ_max) * (1 - ϕ)**(params.α - 1)  # torch.Size([y, 1]), a scalar for each y

    # Gradient of the velocity field (∇U)
    dUx_dy_ = grad(y_, u_)  # torch.Size([y, 1])
    U_grad_ = torch.stack([
        torch.cat([zero, dUx_dy_, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Strain rate tensor (E)
    E_ = 0.5 * (U_grad_ + U_grad_.transpose(1, 2))  # torch.Size([y, 3, 3]), a matrix for each y

    # Shear rate tensor (γ̇)
    γ̇_ = torch.sqrt(2 * torch.sum(E_ * E_, dim=(1, 2))).unsqueeze(1)  # torch.Size([y, 1])

    # Lift force (L)
    γ̇ = γ̇_ * 2 * params.u_max / params.H  # dimensionalize for calculating it
    left_wall = 3 * params.η0 * γ̇ / (4 * torch.pi * ((params.H/2)*(y_ + 1) + params.H0)**params.β) * params.frv
    right_wall = 3 * params.η0 * γ̇ / (4 * torch.pi * ((params.H/2)*(1 - y_) + params.H0)**params.β) * params.frv
    scale_L = (params.H ** 2) / (2 * params.η0 * params.u_max)  # nondimensionalize after calculating it 
    L = torch.stack([
        torch.cat([zero], dim=1),
        torch.cat([scale_L * (left_wall - right_wall)], dim=1),
        torch.cat([zero], dim=1)
    ], dim=1)
    
    # Diagonal tensor of the SBM (Q)
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, params.λ2, 0.0], [0.0, 0.0, params.λ3]], device=config.DEVICE).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Non-local shear rate tensor
    γ̇NL_ = params.ε * params.H / 2

    # Particle normal stress diagonal tensor (Σₙₙᵖ)
    Σpnn_ = ηN(ϕ_).view(-1, 1, 1) * (γ̇_.unsqueeze(1) + γ̇NL_) * Q  # torch.Size([y, 3, 3]), a matrix for each y

    # Oriented particle stress tensor (Σᵖ)
    Σp_ = -Σpnn_ + (2 * ηp(ϕ_).view(-1, 1, 1) * E_)  # torch.Size([y, 3, 3]), a matrix for each y

    # Divergence of oriented particle stress tensor (∇⋅Σᵖ)
    dΣpxy_dy_ = grad(y_, Σp_[:, 0, 1])  # torch.Size([y, 1])
    dΣpyy_dy_ = grad(y_, Σp_[:, 1, 1])  # torch.Size([y, 1])
    Σp_div_ = torch.stack([
        torch.cat([zero + dΣpxy_dy_ + zero], dim=1),
        torch.cat([zero + dΣpyy_dy_ + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y

    # Migration flux (J)
    J_ = - (2 * A**2 / 9) * f(ϕ_).unsqueeze(1) * (Σp_div_ + ϕ_.view(-1, 1, 1) * L)  # torch.Size([y, 3, 1])

    # Divergence of migration flux (∇⋅J)
    dJx_dx_ = dJz_dz_ = zero
    dJy_dy_ = grad(y_, J_[:, 1, 0])  # torch.Size([y, 1])
    J_div_ = dJx_dx_ + dJy_dy_ + dJz_dz_  # torch.Size([y, 1])

    # Identity matrix (I)
    I = torch.eye(3, device=config.DEVICE).repeat(y_.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid phase stress (Σᶠ)
    Σf_ = - p_ * I + 2 * E_

    # Total stress (Σ)
    Σ_ = Σp_ + Σf_

    # Suspension momentum balance (∇⋅Σ)
    dΣxy_dy_ = grad(y_, Σ_[:, 0, 1]) - params.dp_dx  # torch.Size([y, 1])
    dΣyy_dy_ = grad(y_, Σ_[:, 1, 1])  # torch.Size([y, 1])

    return J_div_, dΣxy_dy_, dΣyy_dy_

def ϕ_bulk_loss(trials, params):  # IC ensures mean(ϕ) never changes
    # Initialize
    y_ = params.y_coll_
    u_ = trials.u_trial(y_) 
    ϕ  = trials.ϕ_trial(y_)

    # Mass conservation error calculation
    ϕ_bulk_term = torch.sum(ϕ * u_) / torch.sum(u_) - params.ϕ_bulk 

    return ϕ_bulk_term

def ϕ_symmetry_loss(trials, params):  # ensures ϕ is symmetric along centerflow axis
    # Initialize
    y_ = params.y_coll_

    # Symmetry error calculation 
    ϕ_symmetry_term = trials.ϕ_trial(y_) - trials.ϕ_trial(-y_)

    return ϕ_symmetry_term

def total_loss(trials, params):  # combining losses
    # Call functions only when needed 
    J_div_, dΣxy_dy_, dΣyy_dy_  = physics_loss(trials, params)     # torch.Size([y_coll_.shape[0], 1]) each
    ϕ_bulk_term                 = ϕ_bulk_loss(trials, params)      # torch.Size([1])
    ϕ_symmetry_term             = ϕ_symmetry_loss(trials, params)  # torch.Size([1])
    u_data_term                 = u_data_loss(trials, params)      # torch.Size([y_data_.shape[0], 1])

    # Individual losses
    ℒ_J =        torch.mean(params.mask(params.λ_J)         * J_div_**2)
    ℒ_Σxy =      torch.mean(params.mask(params.λ_Σxy)       * dΣxy_dy_**2)
    ℒ_Σyy =      torch.mean(params.mask(params.λ_Σxy)       * dΣyy_dy_**2)
    ℒ_mass =     torch.mean(params.mask(params.λ_mass)      * ϕ_bulk_term**2)
    ℒ_symmetry = torch.mean(params.mask(params.λ_symmetry)  * ϕ_symmetry_term**2)
    ℒ_data =     torch.mean(params.mask(params.λ_data)      * u_data_term**2)
    # NOTE: ℒ_mass does not need mean(...), as it is already scalars, but for the the sake of code consistiency, they are

    # Sum 
    ℒ   = ℒ_J + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_symmetry + ℒ_data
    ℒ_no_weights = (
        torch.mean(J_div_**2) +
        torch.mean(dΣxy_dy_**2) + 
        torch.mean(dΣyy_dy_**2) + 
        torch.mean(ϕ_bulk_term**2) + 
        torch.mean(ϕ_symmetry_term**2) + 
        torch.mean(u_data_term**2)
        )
    ℒ_individual = [ℒ_J, ℒ_Σxy, ℒ_Σyy, ℒ_mass, ℒ_symmetry, ℒ_data]

    return ℒ, ℒ_no_weights, ℒ_individual
