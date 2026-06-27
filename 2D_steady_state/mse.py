import torch
import config
import geometry
import loss

def profileVisualizationField(field=None):
    if field is None:
        field = getattr(config, "PROFILE_VISUALIZATION_FIELD", "phi")
    field = str(field).strip().lower()
    aliases = {
        "ϕ": "phi",
        "phi": "phi",
        "volume_fraction": "phi",
        "volume fraction": "phi",
        "velocity": "velocity",
        "speed": "velocity",
        "|u|": "velocity",
        "u_magnitude": "velocity",
        "u magnitude": "velocity",
        "lift": "lift",
        "lift_force": "lift",
        "lift force": "lift",
        "lift_magnitude": "lift",
        "lift magnitude": "lift",
        "liftf_magnitude": "lift",
    }
    if field not in aliases:
        raise ValueError("PROFILE_VISUALIZATION_FIELD must be 'phi', 'velocity', or 'lift'.")
    return aliases[field]

def profileFieldLabel(field=None):
    field = profileVisualizationField(field)
    if field == "phi":
        return "ϕ"
    if field == "velocity":
        return "|U| (m/s)"
    return "|LiftF|"

def profileFieldTitle(field=None):
    field = profileVisualizationField(field)
    if field == "phi":
        return "ϕ"
    if field == "velocity":
        return "Velocity"
    return "Lift Force"

# Daughter Branch Local Frame
def daughterBranchFrame(params, which, device, dtype):
    if which == "top":
        theta = torch.as_tensor(params.α1, device=device, dtype=dtype)
        R = torch.as_tensor(params.R1, device=device, dtype=dtype)
        outside_sign = 1.0
    elif which == "bottom":
        theta = torch.as_tensor(-params.α2, device=device, dtype=dtype)
        R = torch.as_tensor(params.R2, device=device, dtype=dtype)
        outside_sign = -1.0
    else:
        raise ValueError("which must be 'top' or 'bottom'.")

    a = torch.tensor([params.LEN_PARENT, 0.0], device=device, dtype=dtype)
    u = torch.stack([torch.cos(theta), torch.sin(theta)])
    n = torch.stack([-torch.sin(theta), torch.cos(theta)])
    outside_start = a + outside_sign * R * n
    inside_start = a - outside_sign * R * n

    return {
        "a": a,
        "u": u,
        "n": n,
        "R": R,
        "outside_sign": outside_sign,
        "outside_start": outside_start,
        "inside_start": inside_start,
        "theta": theta,
    }

def daughterCenterlineIntersectionOffset(params, which, device, dtype):
    frame = daughterBranchFrame(params=params, which=which, device=device, dtype=dtype)
    p = frame["inside_start"]
    r = frame["u"]

    walls = geometry.bifurcationWalls(params=params, device=device, dtype=dtype)
    other_wall_name = "bottom_plus" if which == "top" else "top_minus"
    other_wall = walls[other_wall_name]
    q = other_wall["x0"]
    s = other_wall["x1"] - other_wall["x0"]

    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]

    denom = cross2(r, s)
    if torch.abs(denom) <= 10.0 * torch.finfo(dtype).eps:
        raise ValueError("Could not locate the inner bifurcation corner from daughter wall intersection.")

    return cross2(q - p, s) / denom

def segmentLineIntersection(first, second, dtype):
    p = first["x0"]
    r = first["x1"] - first["x0"]
    q = second["x0"]
    s = second["x1"] - second["x0"]

    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]

    denom = cross2(r, s)
    parallel_tol = 10.0 * torch.finfo(dtype).eps * torch.linalg.norm(r) * torch.linalg.norm(s)
    if torch.abs(denom) <= parallel_tol:
        raise ValueError("Could not locate wall intersection for daughter profile.")

    return p + (cross2(q - p, s) / denom) * r

def daughterProfileWallEndpoints(params, which, device, dtype):
    walls = geometry.bifurcationWalls(params=params, device=device, dtype=dtype)

    if which == "top":
        inside = segmentLineIntersection(walls["top_minus"], walls["bottom_plus"], dtype=dtype)
        outside = segmentLineIntersection(walls["parent_plus"], walls["top_plus"], dtype=dtype)
    elif which == "bottom":
        inside = segmentLineIntersection(walls["bottom_plus"], walls["top_minus"], dtype=dtype)
        outside = segmentLineIntersection(walls["parent_minus"], walls["bottom_minus"], dtype=dtype)
    else:
        raise ValueError("which must be 'top' or 'bottom'.")

    return inside, outside

def parentProfileWallEndpoints(params, device, dtype):
    walls = geometry.bifurcationWalls(params=params, device=device, dtype=dtype)
    bottom = segmentLineIntersection(walls["parent_minus"], walls["bottom_minus"], dtype=dtype)
    top = segmentLineIntersection(walls["parent_plus"], walls["top_plus"], dtype=dtype)
    return bottom, top

def parentProfileLine(params, eta=None, n_profile_pts=None, device=None, dtype=torch.float32):
    if device is None:
        device = params.S.device

    R = torch.as_tensor(params.R0, device=device, dtype=dtype)
    bottom_point, top_point = parentProfileWallEndpoints(params=params, device=device, dtype=dtype)
    center = 0.5 * (bottom_point + top_point)

    if eta is None:
        if n_profile_pts is None:
            raise ValueError("Provide either eta or n_profile_pts.")
        eta = torch.linspace(-R, R, int(n_profile_pts), device=device, dtype=dtype)
    else:
        eta = torch.as_tensor(eta, device=device, dtype=dtype).reshape(-1)
        eta = torch.clamp(eta, min=-R, max=R)

    profile_fraction = (eta + R) / (2.0 * R)
    pts = bottom_point.unsqueeze(0) + profile_fraction.unsqueeze(1) * (top_point - bottom_point).unsqueeze(0)

    return {
        "branch": "parent",
        "points": pts,
        "center": center,
        "eta": eta,
        "R": R,
        "bottom_point": bottom_point,
        "top_point": top_point,
    }

# Daughter Branch Cross-Section Profile
def daughterProfileLine(params, which, dghr_start, eta=None, n_profile_pts=None, device=None, dtype=torch.float32):
    if device is None:
        device = params.S.device

    frame = daughterBranchFrame(params=params, which=which, device=device, dtype=dtype)
    a = frame["a"]
    u = frame["u"]
    n = frame["n"]
    R = frame["R"]
    outside_sign = frame["outside_sign"]
    d = torch.as_tensor(dghr_start, device=device, dtype=dtype)
    use_skewed_profile = params.dghr_skew
    profile_reference = str(getattr(params, "dghr_profile_reference", "branch_origin")).strip().lower()

    if use_skewed_profile:
        inner_corner, outer_bend = daughterProfileWallEndpoints(params=params, which=which, device=device, dtype=dtype)
        inside_point = inner_corner + d * u
        outside_point = outer_bend + d * u
    else:
        if profile_reference in ("v_corner", "inner_corner", "corner"):
            axial_offset = daughterCenterlineIntersectionOffset(
                params=params,
                which=which,
                device=device,
                dtype=dtype,
            )
            centerline_start = a + axial_offset * u
        elif profile_reference in ("branch_origin", "daughter_origin", "origin"):
            centerline_start = a
        else:
            raise ValueError("daughter_profile_start_reference must be 'v_corner' or 'branch_origin'.")
        centerline_point = centerline_start + d * u
        inside_point = centerline_point - outside_sign * R * n
        outside_point = centerline_point + outside_sign * R * n

    center = 0.5 * (inside_point + outside_point)
    chord_length = torch.linalg.norm(outside_point - inside_point)

    if eta is None:
        if n_profile_pts is None:
            raise ValueError("Provide either eta or n_profile_pts.")
        if use_skewed_profile:
            eta = torch.linspace(0.0, chord_length, int(n_profile_pts), device=device, dtype=dtype)
        else:
            eta = torch.linspace(-R, R, int(n_profile_pts), device=device, dtype=dtype)
    else:
        eta = torch.as_tensor(eta, device=device, dtype=dtype).reshape(-1)
        if use_skewed_profile:
            eta = torch.clamp(eta, min=0.0, max=chord_length)
        else:
            eta = torch.clamp(eta, min=-R, max=R)

    if use_skewed_profile:
        profile_fraction = eta / chord_length.clamp_min(torch.finfo(dtype).eps)
    else:
        profile_fraction = (eta + R) / (2.0 * R)
    pts = inside_point.unsqueeze(0) + profile_fraction.unsqueeze(1) * (outside_point - inside_point).unsqueeze(0)

    return {
        "branch": which,
        "points": pts,
        "center": center,
        "eta": eta,
        "u": u,
        "n": n,
        "R": R,
        "outside_point": outside_point,
        "inside_point": inside_point,
        "dghr_start": d,
        "dghr_skewed": use_skewed_profile,
    }

def sampledProfileFromOpenFOAM(branch, xy, eta, arc_length, mask, device, dtype):
    mask = mask.to(device=device, dtype=torch.bool)
    points = xy.to(device=device, dtype=dtype)[mask]
    if points.shape[0] == 0:
        raise ValueError(f"No valid {branch} OpenFOAM profile samples were found.")

    eta_values = torch.as_tensor(eta, device=device, dtype=dtype).reshape(-1)[mask]
    arc_values = torch.as_tensor(arc_length, device=device, dtype=dtype).reshape(-1, 1)[mask]
    center = torch.mean(points, dim=0)

    return {
        "branch": branch,
        "points": points,
        "center": center,
        "eta": eta_values,
        "openfoam_sampled": True,
    }, arc_values

def _finiteProfileMask(*values, device):
    mask = None
    for value in values:
        value_mask = torch.isfinite(torch.as_tensor(value, device=device).reshape(-1))
        mask = value_mask if mask is None else mask & value_mask
    if mask is None:
        raise ValueError("At least one profile value is required to build a mask.")
    return mask

def _maskedProfile(profile, arc_length, mask, device, dtype):
    mask = mask.to(device=device, dtype=torch.bool)
    points = profile["points"][mask]
    if points.shape[0] == 0:
        raise ValueError(f"No valid {profile['branch']} profile samples were found.")

    masked_profile = dict(profile)
    masked_profile["points"] = points
    masked_profile["center"] = torch.mean(points, dim=0)
    masked_profile["eta"] = profile["eta"][mask]
    return masked_profile, torch.as_tensor(arc_length, device=device, dtype=dtype).reshape(-1, 1)[mask]

def _parentProfileFromParams(params, mask, device, dtype):
    profile = parentProfileLine(
        params=params,
        eta=params.prnt_eta,
        device=device,
        dtype=dtype,
    )
    return _maskedProfile(
        profile=profile,
        arc_length=params.prnt_arc,
        mask=mask,
        device=device,
        dtype=dtype,
    )

def _daughterProfileFromParams(params, mask, device, dtype):
    profile_source = str(getattr(params, "dghr_profile_source", "geometry")).strip().lower()
    if profile_source in ("csv", "csv_points", "openfoam", "openfoam_points"):
        return sampledProfileFromOpenFOAM(
            branch="top",
            xy=params.dghr_xy_local,
            eta=params.dghr_eta,
            arc_length=params.dghr_arc,
            mask=mask,
            device=device,
            dtype=dtype,
        )
    if profile_source != "geometry":
        raise ValueError("daughter_profile_source must be 'geometry' or 'csv_points'.")

    profile = daughterProfileLine(
        params=params,
        which="top",
        dghr_start=params.dghr_start,
        eta=params.dghr_eta,
        device=device,
        dtype=dtype,
    )
    return _maskedProfile(
        profile=profile,
        arc_length=params.dghr_arc,
        mask=mask,
        device=device,
        dtype=dtype,
    )

# Daughter Phi Profile Data Loss
def daughterPhiProfileMSE(trials, params, array, device):
    daughter_profile = daughterPhiProfileData(trials=trials, params=params, array=array, device=device)
    ϕ_term = torch.mean((daughter_profile["ϕ_pred"] - daughter_profile["ϕ_data"])**2)
    return ϕ_term

def parentPhiProfileMSE(trials, params, array, device):
    parent_profile = parentPhiProfileData(trials=trials, params=params, array=array, device=device)
    ϕ_term = torch.mean((parent_profile["ϕ_pred"] - parent_profile["ϕ_data"])**2)
    return ϕ_term

def parentPhiProfileData(trials, params, array, device):
    dtype = array["full_array"].dtype
    mask = _finiteProfileMask(params.ϕ_data_prnt, params.prnt_arc, device=device)
    profile, arc_length = _parentProfileFromParams(params=params, mask=mask, device=device, dtype=dtype)

    xy_prof = profile["points"]
    xy_prof_ = xy_prof / params.S
    _, _, _, ϕ_pred = trials.all_trials(xy_prof_)
    ϕ_data = torch.as_tensor(params.ϕ_data_prnt, device=device, dtype=dtype).reshape(-1, 1)[mask]

    return {
        "profile": profile,
        "arc_length": arc_length,
        "ϕ_data": ϕ_data,
        "ϕ_pred": ϕ_pred,
    }

def daughterPhiProfileData(trials, params, array, device):
    dtype = array["full_array"].dtype
    mask = _finiteProfileMask(params.ϕ_data_dghr, params.dghr_arc, device=device)
    mask = mask & params.dghr_valid_mask.to(device=device, dtype=torch.bool)
    profile, arc_length = _daughterProfileFromParams(params=params, mask=mask, device=device, dtype=dtype)

    xy_prof = profile["points"]
    xy_prof_ = xy_prof / params.S
    _, _, _, ϕ_pred = trials.all_trials(xy_prof_)
    ϕ_data = torch.as_tensor(params.ϕ_data_dghr, device=device, dtype=dtype).reshape(-1, 1)[mask]

    return {
        "profile": profile,
        "arc_length": arc_length,
        "ϕ_data": ϕ_data,
        "ϕ_pred": ϕ_pred,
    }

def parentVelocityProfileMSE(trials, params, array, device):
    parent_profile = parentVelocityProfileData(trials=trials, params=params, array=array, device=device)
    velocity_term = torch.mean((parent_profile["velocity_pred"] - parent_profile["velocity_data"])**2)
    return velocity_term

def daughterVelocityProfileMSE(trials, params, array, device):
    daughter_profile = daughterVelocityProfileData(trials=trials, params=params, array=array, device=device)
    velocity_term = torch.mean((daughter_profile["velocity_pred"] - daughter_profile["velocity_data"])**2)
    return velocity_term

def _velocityMagnitudePrediction(trials, params, xy_prof):
    xy_prof_ = xy_prof / params.S
    u_, v_, _, _ = trials.all_trials(xy_prof_)
    return params.u_max * torch.sqrt(u_**2 + v_**2)

def liftMagnitudePrediction(trials, params, array, xy_prof):
    dtype = xy_prof.dtype
    device = xy_prof.device

    xy_req = xy_prof.detach().requires_grad_(True)
    xy_prof_ = xy_req / params.S
    x_ = xy_prof_[:, 0:1]
    zero = torch.zeros_like(x_)

    u_, v_, _, ϕ_ = trials.all_trials(xy_prof_)
    u_grad_ = loss.grad(xy_prof_, u_)
    v_grad_ = loss.grad(xy_prof_, v_)
    du_dx_, du_dy_ = u_grad_[:, 0:1], u_grad_[:, 1:2]
    dv_dx_, dv_dy_ = v_grad_[:, 0:1], v_grad_[:, 1:2]

    U_grad_ = torch.stack([
        torch.cat([du_dx_, du_dy_, zero], dim=1),
        torch.cat([dv_dx_, dv_dy_, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1),
    ], dim=1)
    E_ = 0.5 * (U_grad_ + U_grad_.transpose(1, 2))
    γ̇_ = torch.sqrt(2 * torch.sum(E_ * E_, dim=(1, 2)) + loss.EPS).unsqueeze(1)
    γ̇ = γ̇_ * params.u_max / params.S

    dist_to_wall, dir_to_wall = geometry.nearestSideWallGeometry(
        points=xy_req,
        params=params,
        array=array,
        device=device,
        dtype=dtype,
    )

    def f(ϕ):
        return (1 - ϕ / params.ϕ_max) * (1 - ϕ)**(params.α - 1)

    lift_vec = (
        (
            (3 * params.η0 * γ̇) / (4 * torch.pi) * params.frv
            * dir_to_wall / (dist_to_wall + params.H0)**params.β
        )
        * ϕ_
        * (2 * params.a**2 / (9 * params.η0))
        * f(ϕ_)
    )
    return torch.sqrt(torch.sum(lift_vec**2, dim=1, keepdim=True))

def parentVelocityProfileData(trials, params, array, device):
    dtype = array["full_array"].dtype
    mask = _finiteProfileMask(params.U_data_prnt, params.prnt_arc, device=device)
    profile, arc_length = _parentProfileFromParams(params=params, mask=mask, device=device, dtype=dtype)

    velocity_pred = _velocityMagnitudePrediction(trials=trials, params=params, xy_prof=profile["points"])
    velocity_data = torch.as_tensor(params.U_data_prnt, device=device, dtype=dtype).reshape(-1, 1)[mask]

    return {
        "profile": profile,
        "arc_length": arc_length,
        "velocity_data": velocity_data,
        "velocity_pred": velocity_pred,
    }

def daughterVelocityProfileData(trials, params, array, device):
    dtype = array["full_array"].dtype
    mask = _finiteProfileMask(params.U_data_dghr, params.dghr_arc, device=device)
    mask = mask & params.dghr_valid_mask.to(device=device, dtype=torch.bool)
    profile, arc_length = _daughterProfileFromParams(params=params, mask=mask, device=device, dtype=dtype)

    velocity_pred = _velocityMagnitudePrediction(trials=trials, params=params, xy_prof=profile["points"])
    velocity_data = torch.as_tensor(params.U_data_dghr, device=device, dtype=dtype).reshape(-1, 1)[mask]

    return {
        "profile": profile,
        "arc_length": arc_length,
        "velocity_data": velocity_data,
        "velocity_pred": velocity_pred,
    }

def parentLiftProfileMSE(trials, params, array, device):
    parent_profile = parentLiftProfileData(trials=trials, params=params, array=array, device=device)
    lift_term = torch.mean((parent_profile["lift_mag_pred"] - parent_profile["lift_mag_data"])**2)
    return lift_term

def parentLiftProfileData(trials, params, array, device):
    dtype = array["full_array"].dtype
    lift_mag_all = torch.as_tensor(params.L_mag_prnt, device=device, dtype=dtype).reshape(-1, 1)
    mask = (
        params.L_mag_mask_prnt.to(device=device, dtype=torch.bool)
        & torch.isfinite(lift_mag_all.reshape(-1))
        & torch.isfinite(torch.as_tensor(params.prnt_arc, device=device, dtype=dtype).reshape(-1))
    )
    profile, arc_length = _parentProfileFromParams(params=params, mask=mask, device=device, dtype=dtype)

    lift_mag_pred = liftMagnitudePrediction(
        trials=trials,
        params=params,
        array=array,
        xy_prof=profile["points"],
    )
    lift_mag_data = lift_mag_all[mask]

    finite_mask = torch.isfinite(lift_mag_data) & torch.isfinite(lift_mag_pred) & torch.isfinite(arc_length)
    keep = finite_mask.reshape(-1)
    if not torch.any(keep):
        raise ValueError("No finite valid parent lift-force samples were found.")

    masked_profile = dict(profile)
    masked_profile["points"] = profile["points"][keep]
    masked_profile["eta"] = profile["eta"][keep]
    masked_profile["center"] = torch.mean(masked_profile["points"], dim=0)

    return {
        "profile": masked_profile,
        "arc_length": arc_length[keep],
        "lift_mag_data": lift_mag_data[keep],
        "lift_mag_pred": lift_mag_pred[keep],
    }

def parentProfileMSE(trials, params, array, device, field=None):
    field = profileVisualizationField(field)
    if field == "velocity":
        return parentVelocityProfileMSE(trials=trials, params=params, array=array, device=device)
    if field == "lift":
        return parentLiftProfileMSE(trials=trials, params=params, array=array, device=device)
    return parentPhiProfileMSE(trials=trials, params=params, array=array, device=device)

def daughterProfileMSE(trials, params, array, device, field=None):
    field = profileVisualizationField(field)
    if field == "velocity":
        return daughterVelocityProfileMSE(trials=trials, params=params, array=array, device=device)
    if field == "lift":
        return daughterLiftProfileMSE(trials=trials, params=params, array=array, device=device)
    return daughterPhiProfileMSE(trials=trials, params=params, array=array, device=device)

def parentProfileData(trials, params, array, device, field=None):
    field = profileVisualizationField(field)
    if field == "velocity":
        return parentVelocityProfileData(trials=trials, params=params, array=array, device=device)
    if field == "lift":
        return parentLiftProfileData(trials=trials, params=params, array=array, device=device)
    return parentPhiProfileData(trials=trials, params=params, array=array, device=device)

def daughterProfileData(trials, params, array, device, field=None):
    field = profileVisualizationField(field)
    if field == "velocity":
        return daughterVelocityProfileData(trials=trials, params=params, array=array, device=device)
    if field == "lift":
        return daughterLiftProfileData(trials=trials, params=params, array=array, device=device)
    return daughterPhiProfileData(trials=trials, params=params, array=array, device=device)

def daughterLiftProfileData(trials, params, array, device):
    dtype = array["full_array"].dtype

    lift_mag_all = torch.as_tensor(params.L_mag_dghr, device=device, dtype=dtype).reshape(-1, 1)
    base_mask = (
        params.L_mag_mask.to(device=device, dtype=torch.bool)
        & torch.isfinite(lift_mag_all.reshape(-1))
        & torch.isfinite(torch.as_tensor(params.dghr_arc, device=device, dtype=dtype).reshape(-1))
    )
    profile, arc_length = _daughterProfileFromParams(params=params, mask=base_mask, device=device, dtype=dtype)

    lift_mag_pred = liftMagnitudePrediction(
        trials=trials,
        params=params,
        array=array,
        xy_prof=profile["points"],
    )
    lift_mag_data = lift_mag_all[base_mask]

    finite_mask = torch.isfinite(lift_mag_data) & torch.isfinite(lift_mag_pred) & torch.isfinite(arc_length)
    keep = finite_mask.reshape(-1)
    if not torch.any(keep):
        raise ValueError("No finite valid daughter lift-force samples were found.")

    masked_profile = dict(profile)
    masked_profile["points"] = profile["points"][keep]
    masked_profile["eta"] = profile["eta"][keep]
    masked_profile["center"] = torch.mean(masked_profile["points"], dim=0)

    return {
        "profile": masked_profile,
        "arc_length": arc_length[keep],
        "lift_mag_data": lift_mag_data[keep],
        "lift_mag_pred": lift_mag_pred[keep],
        "valid_mask": keep,
    }

def daughterLiftProfileMSE(trials, params, array, device):
    daughter_profile = daughterLiftProfileData(trials=trials, params=params, array=array, device=device)
    lift_term = torch.mean((daughter_profile["lift_mag_pred"] - daughter_profile["lift_mag_data"])**2)
    return lift_term

def daughterProfileForPlot(params, array, device):
    del array
    dtype = params.S.dtype
    mask = _finiteProfileMask(params.dghr_arc, device=device)
    mask = mask & params.dghr_valid_mask.to(device=device, dtype=torch.bool)
    profile, _ = _daughterProfileFromParams(params=params, mask=mask, device=device, dtype=dtype)
    return profile

def parentProfileForPlot(params, array, device):
    del array

    dtype = params.S.dtype
    mask = _finiteProfileMask(params.prnt_arc, device=device)
    profile, _ = _parentProfileFromParams(params=params, mask=mask, device=device, dtype=dtype)
    return profile
