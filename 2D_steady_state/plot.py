import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import config
import geometry
import paths
from mse import (
    daughterProfileData,
    daughterProfileForPlot,
    liftMagnitudePrediction,
    parentProfileData,
    parentProfileForPlot,
    profileFieldLabel,
    profileFieldTitle,
    profileVisualizationField,
)

class History:
    def __init__(self):
        self.epochs = []
        self.total = []
        self.total_raw = []
        self.components = []
        self.ϕ_mse_epochs = []
        self.ϕ_mse = []
        self.parent_ϕ_mse_epochs = []
        self.parent_ϕ_mse = []

    def append(self, epoch, total, total_raw, components, components_raw):
        del components_raw
        self.epochs.append(int(epoch))
        self.total.append(float(total.detach().cpu().item()))
        self.total_raw.append(float(total_raw.detach().cpu().item()))
        self.components.append([float(c.detach().cpu().item()) for c in components])

    def append_ϕ_mse(self, epoch, ϕ_mse):
        self.ϕ_mse_epochs.append(int(epoch))
        self.ϕ_mse.append(float(ϕ_mse))

    def append_parent_ϕ_mse(self, epoch, ϕ_mse):
        self.parent_ϕ_mse_epochs.append(int(epoch))
        self.parent_ϕ_mse.append(float(ϕ_mse))


def _downsample_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
    return points[idx]


def _lift_quiver_data(array, lift_force, max_points, use_unit_direction):
    points = array["full_array"]
    if lift_force.ndim == 3:
        lift_force = lift_force[:, 0:2, 0]
    else:
        lift_force = lift_force[:, 0:2]

    mag = torch.sqrt(torch.sum(lift_force**2, dim=1, keepdim=True))
    finite_mask = (
        torch.isfinite(points).all(dim=1)
        & torch.isfinite(lift_force).all(dim=1)
        & torch.isfinite(mag).reshape(-1)
        & (mag.reshape(-1) > 0.0)
    )
    if not torch.any(finite_mask):
        return None

    pts = points[finite_mask]
    vec = lift_force[finite_mask]
    mag = mag[finite_mask]

    if pts.shape[0] > max_points:
        idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
        pts = pts[idx]
        vec = vec[idx]
        mag = mag[idx]

    if use_unit_direction:
        vec = vec / mag.clamp_min(1e-12)

    return (
        pts[:, 0].detach().cpu().numpy(),
        pts[:, 1].detach().cpu().numpy(),
        vec[:, 0].detach().cpu().numpy(),
        vec[:, 1].detach().cpu().numpy(),
        mag[:, 0].detach().cpu().numpy(),
    )


def _overlay_lift_quiver(ax, array, lift_force, params):
    q = _lift_quiver_data(
        array=array,
        lift_force=lift_force,
        max_points=int(getattr(config, "PLOT_LIFT_MAX_POINTS", 500)),
        use_unit_direction=True,
    )
    if q is None:
        return

    x, y, fx, fy, mag = q
    del mag

    h_scale = float(params.S.detach().cpu().item()) if torch.is_tensor(params.S) else float(params.S)
    q_len = float(getattr(config, "PLOT_LIFT_QUIVER_LEN", 0.1 * h_scale))

    ax.quiver(
        x,
        y,
        fx * q_len,
        fy * q_len,
        color=getattr(config, "PLOT_LIFT_QUIVER_COLOR", "white"),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        width=float(getattr(config, "PLOT_LIFT_QUIVER_WIDTH", 0.0032)),
        headwidth=4.5,
        headlength=6.0,
        headaxislength=5.0,
        alpha=float(getattr(config, "PLOT_LIFT_QUIVER_ALPHA", 0.95)),
        zorder=5,
    )


def _continuous_fields(trials, params, array, device, dtype, profile_field):
    xmin, xmax, ymin, ymax = geometry.globalDomain(params)
    nx = int(getattr(config, "PLOT_GRID_X", 220))
    ny = int(getattr(config, "PLOT_GRID_Y", 160))

    x_lin = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y_lin = np.linspace(ymin, ymax, ny, dtype=np.float32)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="xy")

    xy_np = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    xy = torch.tensor(xy_np, device=device, dtype=dtype)
    mask = geometry.insideBifurcationMask(xy, params)

    speed_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    ϕ_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    lift_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    u_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    v_grid = np.full((ny, nx), np.nan, dtype=np.float32)

    if mask.any():
        with torch.no_grad():
            xy_m = xy[mask]
            xy_m_ = xy_m / (params.S + 1e-12)

            u_, v_, _, ϕ_ = trials.all_trials(xy_m_)
            u = params.u_max * u_
            v = params.u_max * v_
            speed = torch.sqrt(u**2 + v**2)

        u_vals = u.detach().cpu().numpy().reshape(-1)
        v_vals = v.detach().cpu().numpy().reshape(-1)
        speed_vals = speed.detach().cpu().numpy().reshape(-1)
        ϕ_vals = ϕ_.detach().cpu().numpy().reshape(-1)
        mask_np = mask.detach().cpu().numpy().reshape(-1)
        u_grid.reshape(-1)[mask_np] = u_vals
        v_grid.reshape(-1)[mask_np] = v_vals
        speed_grid.reshape(-1)[mask_np] = speed_vals
        ϕ_grid.reshape(-1)[mask_np] = ϕ_vals

        if profileVisualizationField(profile_field) == "lift":
            lift = liftMagnitudePrediction(
                trials=trials,
                params=params,
                array=array,
                xy_prof=xy_m,
            )
            lift_vals = lift.detach().cpu().numpy().reshape(-1)
            lift_grid.reshape(-1)[mask_np] = lift_vals

    return X, Y, speed_grid, ϕ_grid, lift_grid, u_grid, v_grid


def _cap_velocity_quiver(ax, trials, params, array):
    cap_names = ["inlet", "outlet_top", "outlet_bottom"]
    cap_colors = {
        "inlet": "cyan",
        "outlet_top": "lime",
        "outlet_bottom": "orange",
    }
    h_scale = float(params.S.detach().cpu().item()) if torch.is_tensor(params.S) else float(params.S)
    base_len = float(getattr(config, "PLOT_CAP_QUIVER_LEN", 0.06 * h_scale))
    eps = 1e-12
    first_label = {name: True for name in cap_names}

    for name in cap_names:
        if "points_by_segment" not in array or name not in array["points_by_segment"]:
            continue
        pts = array["points_by_segment"][name]
        if pts is None or pts.shape[0] == 0:
            continue

        with torch.no_grad():
            pts_n = pts / (params.S + 1e-12)
            u, v, _, _ = trials.all_trials(pts_n)
            u = u.reshape(-1)
            v = v.reshape(-1)

        mag = torch.sqrt(u**2 + v**2 + eps)
        valid = mag > 1e-10
        if not torch.any(valid):
            continue

        x = pts[valid, 0].detach().cpu().numpy()
        y = pts[valid, 1].detach().cpu().numpy()
        un = (u[valid] / (mag[valid] + eps) * base_len).detach().cpu().numpy()
        vn = (v[valid] / (mag[valid] + eps) * base_len).detach().cpu().numpy()

        label = name if first_label[name] else None
        ax.quiver(
            x,
            y,
            un,
            vn,
            color=cap_colors[name],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="mid",
            width=0.004,
            headwidth=5.0,
            headlength=6.0,
            headaxislength=5.0,
            alpha=0.95,
            label=label,
            zorder=4,
        )
        first_label[name] = False


def _daughter_profile_overlay_data(trials, params, array, field):
    field = profileVisualizationField(field)
    daughter_profile = daughterProfileData(
        trials=trials,
        params=params,
        array=array,
        device=array["interior_array"].device,
        field=field,
    )
    profile = daughter_profile["profile"]
    data_keys = {
        "phi": "ϕ_data",
        "velocity": "velocity_data",
        "lift": "lift_mag_data",
    }
    pred_keys = {
        "phi": "ϕ_pred",
        "velocity": "velocity_pred",
        "lift": "lift_mag_pred",
    }
    data_key = data_keys[field]
    pred_key = pred_keys[field]

    return {
        "branch": profile["branch"],
        "arc_length": daughter_profile["arc_length"].detach().cpu().numpy().reshape(-1),
        "eta": profile["eta"].detach().cpu().numpy().reshape(-1),
        "data": daughter_profile[data_key].detach().cpu().numpy().reshape(-1),
        "pred": daughter_profile[pred_key].detach().cpu().numpy().reshape(-1),
    }


def _parent_profile_overlay_data(trials, params, array, field):
    field = profileVisualizationField(field)
    parent_profile = parentProfileData(
        trials=trials,
        params=params,
        array=array,
        device=array["interior_array"].device,
        field=field,
    )
    profile = parent_profile["profile"]
    data_keys = {
        "phi": "ϕ_data",
        "velocity": "velocity_data",
        "lift": "lift_mag_data",
    }
    pred_keys = {
        "phi": "ϕ_pred",
        "velocity": "velocity_pred",
        "lift": "lift_mag_pred",
    }
    data_key = data_keys[field]
    pred_key = pred_keys[field]

    return {
        "branch": profile["branch"],
        "arc_length": parent_profile["arc_length"].detach().cpu().numpy().reshape(-1),
        "eta": profile["eta"].detach().cpu().numpy().reshape(-1),
        "data": parent_profile[data_key].detach().cpu().numpy().reshape(-1),
        "pred": parent_profile[pred_key].detach().cpu().numpy().reshape(-1),
    }


def _plotResultsSingle(epoch, trials, params, array, history=None, lift_force=None, profile_field=None, save_path=None):
    profile_field = profileVisualizationField(profile_field)
    profile_label = profileFieldLabel(profile_field)
    profile_title = profileFieldTitle(profile_field)
    xy_coll = array["interior_array"].detach()
    pts = _downsample_points(xy_coll, int(getattr(config, "PLOT_MAX_POINTS", 15000)))
    x_pts = pts[:, 0].cpu().numpy().ravel()
    y_pts = pts[:, 1].cpu().numpy().ravel()

    X, Y, speed_grid, ϕ_grid, lift_grid, u_grid, v_grid = _continuous_fields(
        trials=trials,
        params=params,
        array=array,
        device=xy_coll.device,
        dtype=xy_coll.dtype,
        profile_field=profile_field,
    )
    daughter_profile = daughterProfileForPlot(params=params, array=array, device=xy_coll.device)
    parent_profile = parentProfileForPlot(params=params, array=array, device=xy_coll.device)

    daughter_profile_overlay = _daughter_profile_overlay_data(trials=trials, params=params, array=array, field=profile_field)
    parent_profile_overlay = _parent_profile_overlay_data(trials=trials, params=params, array=array, field=profile_field)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    ax_u = axs[0, 0]
    ax_phi = axs[1, 0]
    ax_phi_mse = axs[0, 1]
    ax_phi_overlay = axs[1, 1]
    ax_parent_mse = axs[0, 2]
    ax_parent_overlay = axs[1, 2]

    if np.isfinite(speed_grid).any():
        s0 = ax_u.contourf(X, Y, speed_grid, levels=40, cmap="viridis")
    else:
        s0 = ax_u.scatter(x_pts, y_pts, s=2, c="k", alpha=0.2, linewidths=0)
    ax_u.scatter(x_pts, y_pts, s=2, c="k", alpha=0.08, linewidths=0)
    ax_u.set_title(f"Velocity magnitude at epoch {epoch}")
    ax_u.set_aspect("equal", adjustable="box")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("y")
    ax_u.grid(alpha=0.2)

    # Velocity direction overlay (quiver) on a coarser grid.
    q_skip = max(1, int(getattr(config, "PLOT_QUIVER_SKIP", 8)))
    Xq = X[::q_skip, ::q_skip]
    Yq = Y[::q_skip, ::q_skip]
    Uq = u_grid[::q_skip, ::q_skip]
    Vq = v_grid[::q_skip, ::q_skip]
    q_mask = np.isfinite(Uq) & np.isfinite(Vq)
    if np.any(q_mask):
        # Normalize arrows so they show direction regardless of local speed magnitude.
        Um = Uq[q_mask]
        Vm = Vq[q_mask]
        mag = np.sqrt(Um**2 + Vm**2)
        eps = 1e-12
        Un = Um / np.maximum(mag, eps)
        Vn = Vm / np.maximum(mag, eps)
        # Fixed visible arrow length in data units.
        dx = float(np.abs(X[0, 1] - X[0, 0])) if X.shape[1] > 1 else 1.0
        dy = float(np.abs(Y[1, 0] - Y[0, 0])) if Y.shape[0] > 1 else 1.0
        base = q_skip * max(dx, dy)
        len_factor = float(getattr(config, "PLOT_QUIVER_LEN_FACTOR", 0.9))
        q_len = max(base * len_factor, 1e-6)
        Uplot = Un * q_len
        Vplot = Vn * q_len
        ax_u.quiver(
            Xq[q_mask],
            Yq[q_mask],
            Uplot,
            Vplot,
            color="white",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="mid",
            width=0.0035,
            headwidth=4.5,
            headlength=6.0,
            headaxislength=5.0,
            alpha=0.9,
        )
    # Velocity-direction arrows exactly on inlet/outlet boundary points.
    _cap_velocity_quiver(ax=ax_u, trials=trials, params=params, array=array)
    fig.colorbar(s0, ax=ax_u, fraction=0.046, pad=0.04)
    handles, labels = ax_u.get_legend_handles_labels()
    if len(handles) > 0:
        ax_u.legend(loc="upper right", fontsize=8)

    profile_grids = {
        "phi": ϕ_grid,
        "velocity": speed_grid,
        "lift": lift_grid,
    }
    profile_cmaps = {
        "phi": "plasma",
        "velocity": "viridis",
        "lift": "magma",
    }
    profile_grid = profile_grids[profile_field]
    profile_cmap = profile_cmaps[profile_field]
    if np.isfinite(profile_grid).any():
        s1 = ax_phi.contourf(X, Y, profile_grid, levels=40, cmap=profile_cmap)
    else:
        s1 = ax_phi.scatter(x_pts, y_pts, s=2, c="k", alpha=0.2, linewidths=0)
    ax_phi.scatter(x_pts, y_pts, s=2, c="k", alpha=0.08, linewidths=0)
    ax_phi.set_title(f"{profile_title} at epoch {epoch}")
    ax_phi.set_aspect("equal", adjustable="box")
    ax_phi.set_xlabel("x")
    ax_phi.set_ylabel("y")
    ax_phi.grid(alpha=0.2)
    if daughter_profile is not None:
        prof_xy = daughter_profile["points"].detach().cpu().numpy()
        center_xy = daughter_profile["center"].detach().cpu().numpy()
        ax_phi.plot(
            prof_xy[:, 0],
            prof_xy[:, 1],
            color="white",
            linewidth=2.2,
            alpha=0.95,
            zorder=5,
            label=f"{daughter_profile['branch']} daughter profile",
        )
        ax_phi.scatter(
            prof_xy[:, 0],
            prof_xy[:, 1],
            s=14,
            facecolors="none",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.85,
            zorder=6,
        )
        ax_phi.scatter(
            center_xy[0],
            center_xy[1],
            s=24,
            c="cyan",
            edgecolors="black",
            linewidths=0.5,
            zorder=7,
        )
    if parent_profile is not None:
        prof_xy = parent_profile["points"].detach().cpu().numpy()
        center_xy = parent_profile["center"].detach().cpu().numpy()
        ax_phi.plot(
            prof_xy[:, 0],
            prof_xy[:, 1],
            color="lime",
            linewidth=2.0,
            alpha=0.95,
            zorder=5,
            label="parent profile",
        )
        ax_phi.scatter(
            prof_xy[:, 0],
            prof_xy[:, 1],
            s=14,
            facecolors="none",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.85,
            zorder=6,
        )
        ax_phi.scatter(
            center_xy[0],
            center_xy[1],
            s=24,
            c="lime",
            edgecolors="black",
            linewidths=0.5,
            zorder=7,
        )
    if lift_force is not None:
        _overlay_lift_quiver(ax=ax_phi, array=array, lift_force=lift_force, params=params)
    fig.colorbar(s1, ax=ax_phi, fraction=0.046, pad=0.04)
    handles, labels = ax_phi.get_legend_handles_labels()
    if len(handles) > 0:
        ax_phi.legend(loc="upper right", fontsize=8)

    ax_phi_overlay.set_title(f"Daughter {profile_title} Profile")
    ax_phi_overlay.set_xlabel("arc length")
    ax_phi_overlay.set_ylabel(profile_label)
    ax_phi_overlay.grid(alpha=0.2)
    ax_phi_overlay.plot(
        daughter_profile_overlay["arc_length"],
        daughter_profile_overlay["pred"],
        color="navy",
        linewidth=2.0,
        label="model",
    )
    ax_phi_overlay.scatter(
        daughter_profile_overlay["arc_length"],
        daughter_profile_overlay["data"],
        s=20,
        color="darkorange",
        alpha=0.85,
        label="data_daughter.csv",
        zorder=3,
    )
    ax_phi_overlay.legend(loc="best", fontsize=8)

    ax_phi_mse.set_title(f"Daughter {profile_title} MSE")
    ax_phi_mse.set_xlabel("epoch")
    ax_phi_mse.set_ylabel("MSE")
    ax_phi_mse.set_xscale("log")
    ax_phi_mse.set_yscale("log")
    ax_phi_mse.grid(alpha=0.2, which="both")
    if history is not None and len(history.ϕ_mse_epochs) > 0 and len(history.ϕ_mse) > 0:
        n = min(len(history.ϕ_mse_epochs), len(history.ϕ_mse))
        epochs = np.asarray(history.ϕ_mse_epochs[:n], dtype=np.int32)
        profile_mse = np.asarray(history.ϕ_mse[:n], dtype=np.float64)
        finite_mask = np.isfinite(profile_mse) & (epochs > 0) & (profile_mse > 0.0)
        if np.any(finite_mask):
            ax_phi_mse.plot(epochs[finite_mask], profile_mse[finite_mask], color="crimson", linewidth=1.8)
            ax_phi_mse.scatter(epochs[finite_mask], profile_mse[finite_mask], s=14, color="crimson", alpha=0.85)
        else:
            ax_phi_mse.text(0.5, 0.5, "No positive MSE yet", ha="center", va="center", transform=ax_phi_mse.transAxes)
    else:
        ax_phi_mse.text(0.5, 0.5, "No MSE history yet", ha="center", va="center", transform=ax_phi_mse.transAxes)

    ax_parent_overlay.set_title(f"Parent {profile_title} Profile")
    ax_parent_overlay.set_xlabel("arc length")
    ax_parent_overlay.set_ylabel(profile_label)
    ax_parent_overlay.grid(alpha=0.2)
    ax_parent_overlay.plot(
        parent_profile_overlay["arc_length"],
        parent_profile_overlay["pred"],
        color="darkgreen",
        linewidth=2.0,
        label="model",
    )
    ax_parent_overlay.scatter(
        parent_profile_overlay["arc_length"],
        parent_profile_overlay["data"],
        s=20,
        color="firebrick",
        alpha=0.85,
        label="data_parent.csv",
        zorder=3,
    )
    ax_parent_overlay.legend(loc="best", fontsize=8)

    ax_parent_mse.set_title(f"Parent {profile_title} MSE")
    ax_parent_mse.set_xlabel("epoch")
    ax_parent_mse.set_ylabel("MSE")
    ax_parent_mse.set_xscale("log")
    ax_parent_mse.set_yscale("log")
    ax_parent_mse.grid(alpha=0.2, which="both")
    if history is not None and len(history.parent_ϕ_mse_epochs) > 0 and len(history.parent_ϕ_mse) > 0:
        n = min(len(history.parent_ϕ_mse_epochs), len(history.parent_ϕ_mse))
        epochs = np.asarray(history.parent_ϕ_mse_epochs[:n], dtype=np.int32)
        parent_profile_mse = np.asarray(history.parent_ϕ_mse[:n], dtype=np.float64)
        finite_mask = np.isfinite(parent_profile_mse) & (epochs > 0) & (parent_profile_mse > 0.0)
        if np.any(finite_mask):
            ax_parent_mse.plot(epochs[finite_mask], parent_profile_mse[finite_mask], color="darkgreen", linewidth=1.8)
            ax_parent_mse.scatter(epochs[finite_mask], parent_profile_mse[finite_mask], s=14, color="darkgreen", alpha=0.85)
        else:
            ax_parent_mse.text(0.5, 0.5, "No positive MSE yet", ha="center", va="center", transform=ax_parent_mse.transAxes)
    else:
        ax_parent_mse.text(0.5, 0.5, "No MSE history yet", ha="center", va="center", transform=ax_parent_mse.transAxes)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    
    #plt.show(block=False)
    #plt.pause(0.15)
    #plt.close(fig)


def plotResults(epoch, trials, params, array, history=None, lift_force=None):
    save_steps = int(getattr(config, "SAVE_STEPS", 0))
    should_save = save_steps > 0 and epoch % save_steps == 0

    if should_save:
        results_dir = paths.ROOT / "results"
        for field, filename_field in (
            ("phi", "phi"),
            ("velocity", "velocity"),
            ("lift", "lift_force"),
        ):
            _plotResultsSingle(
                epoch=epoch,
                trials=trials,
                params=params,
                array=array,
                history=history,
                lift_force=lift_force,
                profile_field=field,
                save_path=results_dir / f"plot_epoch_{epoch}_{filename_field}.png",
            )
        return

    _plotResultsSingle(
        epoch=epoch,
        trials=trials,
        params=params,
        array=array,
        history=history,
        lift_force=lift_force,
        profile_field=profileVisualizationField(),
    )
