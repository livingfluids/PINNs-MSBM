import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import geometry
import paths


class History:
    def __init__(self):
        self.epochs = []
        self.total = []
        self.total_raw = []
        self.components = []

    def append(self, epoch, total, total_raw, components, components_raw):
        del components_raw
        self.epochs.append(int(epoch))
        self.total.append(float(total.detach().cpu().item()))
        self.total_raw.append(float(total_raw.detach().cpu().item()))
        self.components.append([float(c.detach().cpu().item()) for c in components])


def _downsample_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
    return points[idx]


def _continuous_fields(trials, params, device, dtype):
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
    phi_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    u_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    v_grid = np.full((ny, nx), np.nan, dtype=np.float32)

    if mask.any():
        with torch.no_grad():
            xy_m = xy[mask]
            xy_m_ = xy_m / (params.S + 1e-12)

            u = params.u_max * trials.u_trial(xy_m_)
            v = params.u_max * trials.v_trial(xy_m_)
            phi = trials.ϕ_trial(xy_m_)
            speed = params.u_max * torch.sqrt(u**2 + v**2)

        u_vals = u.detach().cpu().numpy().reshape(-1)
        v_vals = v.detach().cpu().numpy().reshape(-1)
        speed_vals = speed.detach().cpu().numpy().reshape(-1)
        phi_vals = phi.detach().cpu().numpy().reshape(-1)
        mask_np = mask.detach().cpu().numpy().reshape(-1)
        u_grid.reshape(-1)[mask_np] = u_vals
        v_grid.reshape(-1)[mask_np] = v_vals
        speed_grid.reshape(-1)[mask_np] = speed_vals
        phi_grid.reshape(-1)[mask_np] = phi_vals

    return X, Y, speed_grid, phi_grid, u_grid, v_grid


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
            u = trials.u_trial(pts_n).reshape(-1)
            v = trials.v_trial(pts_n).reshape(-1)

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


def plotResults(epoch, trials, params, array):
    xy_coll = array["interior_array"].detach()
    pts = _downsample_points(xy_coll, int(getattr(config, "PLOT_MAX_POINTS", 15000)))
    x_pts = pts[:, 0].cpu().numpy().ravel()
    y_pts = pts[:, 1].cpu().numpy().ravel()

    X, Y, speed_grid, phi_grid, u_grid, v_grid = _continuous_fields(
        trials=trials,
        params=params,
        device=xy_coll.device,
        dtype=xy_coll.dtype,
    )

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    ax0, ax1 = axs

    if np.isfinite(speed_grid).any():
        s0 = ax0.contourf(X, Y, speed_grid, levels=40, cmap="viridis")
    else:
        s0 = ax0.scatter(x_pts, y_pts, s=2, c="k", alpha=0.2, linewidths=0)
    ax0.scatter(x_pts, y_pts, s=2, c="k", alpha=0.08, linewidths=0)
    ax0.set_title(f"Continuous Speed |u| @ epoch {epoch}")
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.grid(alpha=0.2)

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
        ax0.quiver(
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
    _cap_velocity_quiver(ax=ax0, trials=trials, params=params, array=array)
    fig.colorbar(s0, ax=ax0, fraction=0.046, pad=0.04)
    handles, labels = ax0.get_legend_handles_labels()
    if len(handles) > 0:
        ax0.legend(loc="upper right", fontsize=8)

    if np.isfinite(phi_grid).any():
        s1 = ax1.contourf(X, Y, phi_grid, levels=40, cmap="plasma")
    else:
        s1 = ax1.scatter(x_pts, y_pts, s=2, c="k", alpha=0.2, linewidths=0)
    ax1.scatter(x_pts, y_pts, s=2, c="k", alpha=0.08, linewidths=0)
    ax1.set_title("Continuous Phi Field")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.2)
    fig.colorbar(s1, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_steps = int(getattr(config, "SAVE_STEPS", 0))
    if save_steps > 0 and epoch % save_steps == 0:
        paths.visu_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(paths.visu_dir / f"plot_epoch_{epoch}.png", dpi=220, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.15)
    plt.close(fig)
