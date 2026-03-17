import numpy as np
import torch
import matplotlib.pyplot as plt


def visualizeArray(array):
    def to_numpy_xy(a):
        if torch.is_tensor(a): xy0 = a.detach().cpu().numpy()
        else: xy0 = np.asarray(a)

        if xy0.ndim != 2 or xy0.shape[1] != 2: raise ValueError("array must have shape (N, 2)")
        return xy0

    fig, ax = plt.subplots(figsize=(8, 6))

    if isinstance(array, dict):
        if "coll_array" in array and "points_by_segment" in array:
            coll_xy = to_numpy_xy(array["coll_array"])
            ax.scatter(coll_xy[:, 0], coll_xy[:, 1], s=3, c="black", alpha=0.35, label="collocation")

            color_map = {
                "inlet": "royalblue",
                "outlet_top": "limegreen",
                "outlet_bottom": "darkorange",
            }
            for name, pts0 in array["points_by_segment"].items():
                pts = to_numpy_xy(pts0)
                if pts.shape[0] == 0: continue
                color = color_map[name] if name in color_map else "crimson"
                ax.scatter(pts[:, 0], pts[:, 1], s=9, c=color, alpha=0.95, label=name)
            ax.legend(loc="best")
        elif "coll_array" in array and "boundary_array" in array:
            coll_xy = to_numpy_xy(array["coll_array"])
            bnd_xy = to_numpy_xy(array["boundary_array"])

            ax.scatter(coll_xy[:, 0], coll_xy[:, 1], s=3, c="black", alpha=0.45, label="collocation")
            ax.scatter(bnd_xy[:, 0], bnd_xy[:, 1], s=6, c="crimson", alpha=0.9, label="boundary")
            ax.legend(loc="best")
        elif "full_array" in array and "is_boundary" in array:
            full_xy = to_numpy_xy(array["full_array"])
            if torch.is_tensor(array["is_boundary"]): bmask = array["is_boundary"].detach().cpu().numpy().astype(bool)
            else: bmask = np.asarray(array["is_boundary"], dtype=bool)
            ax.scatter(full_xy[~bmask, 0], full_xy[~bmask, 1], s=3, c="black", alpha=0.45, label="collocation")
            ax.scatter(full_xy[bmask, 0], full_xy[bmask, 1], s=6, c="crimson", alpha=0.9, label="boundary")
            ax.legend(loc="best")
        else: raise ValueError("dict input must contain either (coll_array & boundary_array) or (full_array & is_boundary)")
    else:
        xy = to_numpy_xy(array)
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c="black", alpha=0.7)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Geometry Points")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
