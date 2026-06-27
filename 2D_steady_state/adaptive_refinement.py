import contextlib

import torch

import config
import geometry
from loss import PDELoss


EPS = 1e-12

COMPONENT_CONFIG = {
    "migration": "RAR_WEIGHT_MIGRATION",
    "x_momentum": "RAR_WEIGHT_X_MOMENTUM",
    "y_momentum": "RAR_WEIGHT_Y_MOMENTUM",
    "continuity": "RAR_WEIGHT_CONTINUITY",
    "phi_continuity": "RAR_WEIGHT_PHI_CONTINUITY",
}


def residualBasedAdaptiveRefinement(trials, params, PINN, array, device, epoch):
    if not shouldRefine(epoch=epoch, array=array):
        return array, None

    current_count = int(array["interior_array"].shape[0])
    max_points = _maxInteriorPoints()
    n_add = int(getattr(config, "RAR_ADD_POINTS", 100))
    if max_points is not None:
        n_add = min(n_add, max(0, max_points - current_count))
    if n_add <= 0:
        return array, None

    n_candidates = max(int(getattr(config, "RAR_CANDIDATE_POINTS", 20_000)), n_add)
    candidates = geometry.sampleInteriorPoints(
        params=params,
        device=device,
        n_points=n_candidates,
        dtype=array["full_array"].dtype,
        proposal_factor=float(getattr(config, "RAR_PROPOSAL_FACTOR", 8)),
    )

    components = _candidateResidualComponents(
        trials=trials,
        params=params,
        PINN=PINN,
        base_array=array,
        candidates=candidates,
        device=device,
    )
    residuals = _residualIndicator(components)
    residuals = torch.nan_to_num(residuals, nan=0.0, posinf=1e12, neginf=0.0).clamp_min(0.0)

    if residuals.numel() == 0:
        return array, None

    max_residual = float(torch.max(residuals).item())
    if max_residual <= float(getattr(config, "RAR_MIN_RESIDUAL", 0.0)):
        return array, None

    selected_idx = _selectCandidateIndices(residuals=residuals, n_add=n_add)
    selected_points = candidates[selected_idx.to(device=device)].detach()
    selected_residuals = residuals[selected_idx]

    interior_array = torch.cat([array["interior_array"].detach(), selected_points], dim=0)
    array = geometry.rebuildBifurcationArrayWithInterior(
        params=params,
        device=device,
        array=array,
        interior_array=interior_array,
    )

    stats = {
        "method": str(getattr(config, "RAR_METHOD", "rar-d")),
        "added": int(selected_points.shape[0]),
        "interior_before": current_count,
        "interior_after": int(array["interior_array"].shape[0]),
        "candidate_points": int(candidates.shape[0]),
        "residual_mean": float(torch.mean(residuals).item()),
        "residual_max": max_residual,
        "selected_residual_mean": float(torch.mean(selected_residuals).item()),
        "selected_residual_min": float(torch.min(selected_residuals).item()),
    }
    return array, stats


def shouldRefine(epoch, array):
    if not bool(getattr(config, "RAR_ENABLED", False)):
        return False
    if int(epoch) < int(getattr(config, "RAR_START_EPOCH", 1_000)):
        return False
    if int(epoch) >= int(getattr(config, "EPOCHS", 0)):
        return False

    every = max(1, int(getattr(config, "RAR_EVERY", 1_000)))
    if int(epoch) % every != 0:
        return False

    max_points = _maxInteriorPoints()
    if max_points is not None and int(array["interior_array"].shape[0]) >= max_points:
        return False

    return True


def _candidateResidualComponents(trials, params, PINN, base_array, candidates, device):
    batch_size = max(1, int(getattr(config, "RAR_BATCH_SIZE", 1024)))
    chunks = {name: [] for name in COMPONENT_CONFIG}
    was_training = PINN.training

    with _disableParameterGrad(PINN):
        PINN.eval()
        try:
            for start in range(0, candidates.shape[0], batch_size):
                batch = candidates[start:start + batch_size]
                batch_array = _candidateArray(base_array=base_array, points=batch)
                migration, x_momentum, y_momentum, continuity, phi_continuity, _ = PDELoss(
                    trials=trials,
                    params=params,
                    array=batch_array,
                    device=device,
                )
                batch_components = {
                    "migration": migration,
                    "x_momentum": x_momentum,
                    "y_momentum": y_momentum,
                    "continuity": continuity,
                    "phi_continuity": phi_continuity,
                }
                for name, values in batch_components.items():
                    values = values.detach().reshape(-1).cpu()
                    values = torch.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
                    chunks[name].append(values)
        finally:
            if was_training:
                PINN.train()

    return {
        name: torch.cat(values, dim=0) if len(values) > 0 else torch.empty(0)
        for name, values in chunks.items()
    }


def _residualIndicator(components):
    first = next(iter(components.values()))
    residual_sq = torch.zeros_like(first)
    normalize = bool(getattr(config, "RAR_NORMALIZE_COMPONENTS", True))

    for name, values in components.items():
        weight = float(getattr(config, COMPONENT_CONFIG[name], 0.0))
        if weight <= 0.0:
            continue
        values = values.to(dtype=residual_sq.dtype)
        if normalize:
            scale = torch.mean(torch.abs(values)).clamp_min(EPS)
            values = values / scale
        residual_sq = residual_sq + weight * values * values

    return torch.sqrt(residual_sq.clamp_min(0.0))


def _selectCandidateIndices(residuals, n_add):
    n_add = min(int(n_add), int(residuals.numel()))
    method = str(getattr(config, "RAR_METHOD", "rar-d")).strip().lower()

    if method in ("rar-g", "rarg", "greedy", "topk", "rar"):
        return torch.topk(residuals, k=n_add, largest=True).indices

    if method in ("rar-d", "rard", "distribution", "probability", "probabilistic"):
        k = max(float(getattr(config, "RAR_D_K", 2.0)), 0.0)
        c = max(float(getattr(config, "RAR_D_C", 0.0)), 0.0)
        scores = residuals.clamp_min(0.0).pow(k)
        scores = scores / torch.mean(scores).clamp_min(EPS) + c
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0).clamp_min(0.0)

        if int(torch.count_nonzero(scores).item()) < n_add:
            scores = scores + EPS
        if torch.sum(scores) <= EPS:
            scores = torch.ones_like(scores)

        probs = scores / torch.sum(scores)
        return torch.multinomial(probs, num_samples=n_add, replacement=False)

    raise ValueError("Unknown RAR_METHOD. Use 'rar-d' or 'rar-g'.")


def _candidateArray(base_array, points):
    n_points = points.shape[0]
    return {
        "interior_array": points,
        "boundary_array": points.new_empty((0, 2)),
        "boundary_segment_id": torch.empty((0,), device=points.device, dtype=torch.long),
        "full_array": points,
        "is_interior": torch.ones(n_points, device=points.device, dtype=torch.bool),
        "is_boundary": torch.zeros(n_points, device=points.device, dtype=torch.bool),
        "full_segment_id": torch.full((n_points,), -1, device=points.device, dtype=torch.long),
        "segment_names": base_array["segment_names"],
        "boundary_normals": points.new_empty((0, 2)),
        "side_wall_segments": base_array["side_wall_segments"],
    }


def _maxInteriorPoints():
    value = getattr(config, "RAR_MAX_INTERIOR_POINTS", None)
    if value is None:
        return None
    return int(value)


@contextlib.contextmanager
def _disableParameterGrad(module):
    params = list(module.parameters())
    states = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, state in zip(params, states):
            p.requires_grad_(state)
