import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

import config
import plot
from architecture import buildSimplePINN, buildTrials


def choose_device():
    if config.USE_GPU:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_pinn_from_checkpoint(checkpoint, device):
    state_dict = checkpoint["model_state_dict"]
    if "0.B" not in state_dict:
        raise KeyError("Checkpoint missing expected key '0.B' for neuron inference.")
    neurons = state_dict["0.B"].shape[1]

    pinn = buildSimplePINN(neurons=neurons).to(device)
    pinn.load_state_dict(state_dict)
    pinn.eval()
    return pinn


def build_params_from_checkpoint(checkpoint, device):
    saved_params = checkpoint.get("params", {})
    params_tensors = {name: value.to(device) for name, value in saved_params.items()}
    return SimpleNamespace(**params_tensors)


def build_history_from_checkpoint(checkpoint):
    saved_history = checkpoint.get("history", {})
    history = plot.History()
    history.epochs = saved_history.get("epochs", [])
    history.total = saved_history.get("total", [])
    history.total_no_weight = saved_history.get("total_no_weight", [])
    history.individuals = saved_history.get("individuals", [])
    history.individuals_no_weight = saved_history.get("individuals_no_weight", [])
    history_beta_key = "\u03b2s"
    setattr(history, history_beta_key, saved_history.get(history_beta_key, []))
    history.MSEs = saved_history.get("MSEs", [])
    return history


def resolve_checkpoint_path(checkpoint_arg):
    if checkpoint_arg is not None:
        checkpoint_path = checkpoint_arg.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    results_dir = Path(__file__).resolve().parent / "results"
    candidates = sorted(results_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint files found in: {results_dir}")
    return candidates[0]


def render_plot_blocking(epoch, trials, params, history, save_path):
    original_close = plot.plt.close
    try:
        # plotResults auto-closes figures; disable that so the window persists.
        plot.plt.close = lambda *args, **kwargs: None
        plot.plotResults(epoch=epoch, trials=trials, params=params, history=history)
        figure = plot.plt.gcf()
        figure.savefig(save_path, dpi=300, bbox_inches="tight")
        plot.plt.show(block=True)
    finally:
        plot.plt.close = original_close


def main():
    parser = argparse.ArgumentParser(description="Visualize a saved training checkpoint.")
    parser.add_argument("checkpoint", type=Path, nargs="?", default=None, help="Path to a saved .pt checkpoint file.")
    parser.add_argument("--epoch", type=int, default=None, help="Override epoch label used in plot title.")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint_path = Path(__file__).resolve().parent / "results" / "synthetic_data_example_1_learn_beta_6000_inverse.pt"

    device = choose_device()
    checkpoint = load_checkpoint(checkpoint_path, device)

    if "case" in checkpoint:
        config.CASE = checkpoint["case"]

    pinn = build_pinn_from_checkpoint(checkpoint, device)
    params = build_params_from_checkpoint(checkpoint, device)
    history = build_history_from_checkpoint(checkpoint)
    trials = buildTrials(params=params, device=device, PINN=pinn)

    epoch = args.epoch if args.epoch is not None else int(checkpoint.get("epoch", 0))
    image_path = checkpoint_path.with_suffix(".png")
    render_plot_blocking(epoch=epoch, trials=trials, params=params, history=history, save_path=image_path)
    print(f"Rendered plot for checkpoint: {checkpoint_path}")
    print(f"Saved image: {image_path}")


if __name__ == "__main__":
    main()
