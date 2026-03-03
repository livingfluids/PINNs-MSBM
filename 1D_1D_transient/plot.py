import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm


# Plot Results
def plotResults(trials, params, save_dir: Optional[Path] = None, fname: Optional[str] = None, show: bool = True):
    with torch.no_grad():
        save_dir = Path("results") if save_dir is None else Path(save_dir)

        def _cell_edges_from_centers(x: np.ndarray) -> np.ndarray:
            if x.size == 1:
                pad = max(1e-12, float(x[0]) * 0.5)
                return np.array([x[0] - pad, x[0] + pad], dtype=x.dtype)
            dx = np.diff(x)
            mids = x[:-1] + 0.5 * dx
            left = x[0] - 0.5 * dx[0]
            right = x[-1] + 0.5 * dx[-1]
            return np.concatenate(([left], mids, [right]))

        # Data-space grids
        y_data_dim       = ((params.y_data_ + 1.0) / 2.0 * params.H).squeeze(-1)  # [Ny_d]
        t_data           = params.t_data_.squeeze()                                 # log-normalized [0, 1]
        y_data_np        = y_data_dim.cpu().numpy()
        t_data_np        = t_data.cpu().numpy()
        t_edges          = _cell_edges_from_centers(t_data_np)
        y_edges          = _cell_edges_from_centers(y_data_np)

        # Data/model slices on data grid
        u_data           = (params.u_data_ * params.u_max).permute(1, 0, 2).squeeze(-1)  # [Ny_d, Nt_d]
        ϕ_data           = params.ϕ_data_.permute(1, 0, 2).squeeze(-1)                    # [Ny_d, Nt_d]
        u_model_data, _  = trials.u_trial(params.y_data_, params.t_data_)                 # [Ny_d, Nt_d, 1]
        ϕ_model_data, _  = trials.ϕ_trial(params.y_data_, params.t_data_)                 # [Ny_d, Nt_d, 1]
        u_model_data     = u_model_data.squeeze(-1).cpu().numpy()                         # [Ny_d, Nt_d]
        ϕ_model_data     = ϕ_model_data.squeeze(-1).cpu().numpy()                         # [Ny_d, Nt_d]
        ϕ_data_np        = ϕ_data.cpu().numpy()                                            # [Ny_d, Nt_d]
        ϕ_diff           = ϕ_data_np - ϕ_model_data                                        # [Ny_d, Nt_d]
        diff_abs_max     = max(float(np.max(np.abs(ϕ_diff))), 1e-12)
        diff_norm        = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0.0, vmax=diff_abs_max)

        Nt_d             = t_data.shape[0]
        t_indices        = [0, Nt_d // 2, Nt_d - 1] if Nt_d >= 3 else list(range(Nt_d))

        # Figure layout: 3 heatmaps (top), 3 u slices + 3 ϕ slices (bottom)
        fig              = plt.figure(figsize=(18, 8))
        gs               = fig.add_gridspec(2, 12, height_ratios=[1.1, 1.0], hspace=0.4, wspace=0.35)

        # u heatmap
        ax_u             = fig.add_subplot(gs[0, 0:4])
        im_u             = ax_u.pcolormesh(
            t_edges,
            y_edges,
            u_model_data * params.u_max.cpu().numpy(),
            cmap="viridis",
            shading="flat",
        )
        ax_u.set_title("u (model, data-time grid)")
        ax_u.set_xlabel("t (normalized)")
        ax_u.set_ylabel("y [m]")
        fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)

        # ϕ heatmap
        ax_ϕ             = fig.add_subplot(gs[0, 4:8])
        im_ϕ             = ax_ϕ.pcolormesh(
            t_edges,
            y_edges,
            ϕ_model_data,
            cmap="magma",
            shading="flat",
        )
        ax_ϕ.set_title("ϕ (model, data-time grid)")
        ax_ϕ.set_xlabel("t (normalized)")
        ax_ϕ.set_ylabel("y [m]")
        fig.colorbar(im_ϕ, ax=ax_ϕ, fraction=0.046, pad=0.04)

        # ϕ difference heatmap (data - model)
        ax_ϕ_diff        = fig.add_subplot(gs[0, 8:12])
        im_ϕ_diff        = ax_ϕ_diff.pcolormesh(
            t_edges,
            y_edges,
            ϕ_diff,
            cmap="coolwarm",
            norm=diff_norm,
            shading="flat",
        )
        ax_ϕ_diff.set_title("ϕ data - ϕ model")
        ax_ϕ_diff.set_xlabel("t (normalized)")
        ax_ϕ_diff.set_ylabel("y [m]")
        fig.colorbar(im_ϕ_diff, ax=ax_ϕ_diff, fraction=0.046, pad=0.04)

        # Time slices
        for i, t_idx in enumerate(t_indices):
            t_val = t_data[t_idx : t_idx + 1]  # [1]

            # Model on data y-grid at this data time
            u_slice_model, _ = trials.u_trial(params.y_data_, t_val.unsqueeze(1))
            ϕ_slice_model, _ = trials.ϕ_trial(params.y_data_, t_val.unsqueeze(1))

            u_slice_model = u_slice_model.squeeze(-1).cpu().numpy().squeeze()
            ϕ_slice_model = ϕ_slice_model.squeeze(-1).cpu().numpy().squeeze()

            u_slice_data = u_data[:, t_idx].cpu().numpy()
            ϕ_slice_data = ϕ_data[:, t_idx].cpu().numpy()

            ax_u_slice = fig.add_subplot(gs[1, 2 * i : 2 * (i + 1)])
            ax_u_slice.plot(y_data_np, u_slice_data, "ko", markersize=3, label="u data")
            ax_u_slice.plot(y_data_np, u_slice_model * params.u_max.cpu().numpy(), "b-", linewidth=1.5, label="u model")
            ax_u_slice.set_xlabel("y [m]")
            ax_u_slice.set_ylabel("u [m/s]")
            ax_u_slice.set_title(f"u slice t_idx={t_idx}")
            ax_u_slice.grid(True)
            ax_u_slice.legend(fontsize=8)

            ax_ϕ_slice = fig.add_subplot(gs[1, 6 + 2 * i : 8 + 2 * i])
            ax_ϕ_slice.plot(y_data_np, ϕ_slice_data, "ko", markersize=3, label="ϕ data")
            ax_ϕ_slice.plot(y_data_np, ϕ_slice_model, "r-", linewidth=1.5, label="ϕ model")
            ax_ϕ_slice.axhline(params.ϕ_max.item(), color="r", linestyle="--", label=f"ϕ Max = {params.ϕ_max.item():.4f}")
            ax_ϕ_slice.set_xlabel("y [m]")
            ax_ϕ_slice.set_ylabel("ϕ [dimensionless]")
            ax_ϕ_slice.set_title(f"ϕ slice t_idx={t_idx}")
            ax_ϕ_slice.grid(True)
            ax_ϕ_slice.legend(fontsize=8)

        plt.tight_layout()

        save_dir.mkdir(parents=True, exist_ok=True)
        filename = fname or "viz.png"
        fig.savefig(save_dir / filename, dpi=200, bbox_inches="tight")

        if show:
            plt.show(block=False)
            plt.pause(0.5)
        plt.close(fig)

        # 3D overlay: learned ϕ profile + imposed ϕ data on (t, y)
        t_grid_data, y_grid_data = np.meshgrid(t_data_np, y_data_np, indexing="xy")

        fig_3d = plt.figure(figsize=(10, 7))
        ax_3d = fig_3d.add_subplot(111, projection="3d")

        surface = ax_3d.plot_surface(
            t_grid_data,
            y_grid_data,
            ϕ_model_data,
            cmap="viridis",
            alpha=0.75,
            linewidth=0,
            antialiased=True,
        )
        ax_3d.scatter(
            t_grid_data.ravel(),
            y_grid_data.ravel(),
            ϕ_data_np.ravel(),
            c="k",
            s=12,
            alpha=0.9,
            depthshade=False,
        )

        ax_3d.set_xlabel("t (normalized)")
        ax_3d.set_ylabel("y [m]")
        ax_3d.set_zlabel("ϕ [dimensionless]")
        ax_3d.set_title("ϕ(y, t): learned profile + imposed data")
        fig_3d.colorbar(surface, ax=ax_3d, shrink=0.7, pad=0.1, label="ϕ model")
        ax_3d.legend(
            handles=[Line2D([0], [0], marker="o", color="k", linestyle="None", markersize=6, label="ϕ data")],
            loc="upper right",
        )

        save_dir.mkdir(parents=True, exist_ok=True)
        if fname is None:
            fname_3d = "viz_phi_3d.png"
        else:
            stem = Path(fname).stem
            suffix = Path(fname).suffix or ".png"
            fname_3d = f"{stem}_phi_3d{suffix}"
        fig_3d.savefig(save_dir / fname_3d, dpi=200, bbox_inches="tight")

        """if show:
            plt.show(block=False)
            plt.pause(0.5)
        plt.close(fig_3d)"""
