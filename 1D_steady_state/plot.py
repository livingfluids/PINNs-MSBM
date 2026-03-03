import matplotlib.pyplot as plt
import torch
import numpy as np
import paths
from loss import ϕMSE
import config

# Loss History Class
class History:
    def __init__(self):
        self.epochs = []
        self.total = []
        self.total_no_weight = []
        self.individuals = []
        self.individuals_no_weight = []
        self.βs = []
        self.MSEs = []

    def append(self, epoch, ℒ, ℒ_no_weight, ℒ_individuals, ℒ_individuals_no_weight, β, MSE):
        self.epochs.append(epoch)
        self.total.append(ℒ.item())
        self.total_no_weight.append(ℒ_no_weight.item())
        self.individuals.append([ℒ_individual.item() for ℒ_individual in ℒ_individuals])
        self.individuals_no_weight.append([ℒ_individual_no_weight.item() for ℒ_individual_no_weight in ℒ_individuals_no_weight])
        self.βs.append(float(β.detach().cpu().item()))
        self.MSEs.append(float(MSE.detach().cpu().item()))

# Find & Highlight Predicted CFL 
def findCFL(ax, trials, params, n_points=20, threshold=0.1, color='red', alpha=0.12):
    with torch.enable_grad():
        y = torch.linspace(-1.0, 1.0, n_points, device=params.y_coll_.device, dtype=params.y_coll_.dtype).unsqueeze(1)
        y.requires_grad_(True)
        ϕ = trials.ϕ_trial(y)
        dϕ_dy = torch.autograd.grad(ϕ, y, grad_outputs=torch.ones_like(ϕ), retain_graph=False, create_graph=False)[0]

    y_plot = ((y.detach() + 1.0) / 2.0 * params.H).cpu().numpy().flatten()
    grad = dϕ_dy.detach().cpu().numpy().flatten()
    grad_diff = np.abs(np.diff(grad))

    label_used = False
    for i, diff in enumerate(grad_diff):
        if diff > threshold:
            left, right = y_plot[i], y_plot[i + 1]
            ax.axvspan(min(left, right), max(left, right), color=color, alpha=alpha, label=None if label_used else f'|Δϕ\'| > {threshold}')
            label_used = True

# Plot 
def plotResults(epoch, trials, params, history, grad_threshold=0.2, grad_points=30):
    with torch.no_grad():
        # Matplotlib needs CPU numpy arrays
        to_numpy = lambda t: t.detach().cpu().numpy()

        y_plot = ((params.y_coll_ + 1.0) / 2.0 * params.H).cpu().numpy()
        y_plot_data_ = ((params.y_data_ + 1.0) / 2.0 * params.H).cpu().numpy()
        u_plot = (trials.u_trial(params.y_coll_) * params.u_max).cpu().numpy()
        ϕ_plot = trials.ϕ_trial(params.y_coll_).cpu().numpy()

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        ax1 = axs[0][0]
        ax2 = axs[0][1]
        ax3 = axs[0][2]
        ax4 = axs[1][0]
        ax5 = axs[1][1]
        ax6 = axs[1][2]
        panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        for label, ax in zip(panel_labels, [ax1, ax2, ax3, ax4, ax5, ax6]):
            ax.text(
                0.02, 0.98, label,
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=12,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5),
            )

        # u plot
        ax1.plot(y_plot, u_plot, label=f'u at epoch {epoch}')
        ax1.plot(y_plot_data_, to_numpy(params.u_data_ * params.u_max), 'ko', markersize=3)
        ax1.set_xlabel('y [m]')
        ax1.set_ylabel('u [m/s]')
        ax1.legend()
        ax1.grid()

        # ϕ plot
        ax2.plot(y_plot, ϕ_plot, label=f'ϕ at epoch {epoch}')
        ax2.axhline(params.ϕ_max.item(), color='r', linestyle='--', label=f'ϕ Max = {params.ϕ_max.item():.4f}')
        ax2.plot(y_plot_data_, to_numpy(params.ϕ_data_), 'ko', markersize=3)
        ax2.set_xlabel('y [m]')
        ax2.set_ylabel('ϕ [dimensionless]')
        ax2.grid()

        # Beta
        if config.CASE == 'learn cfl':
            if history.MSEs:
                ax3.semilogy(history.MSEs, 'g-', label=f'ϕ MSE = {ϕMSE(trials, params).item():.4f}')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('ϕ MSE')
                ax3.legend()
                ax3.grid(True)
        elif config.CASE == 'learn beta':
            if history.βs:
                ax3.semilogy(history.βs, 'g-', label=f'β = {params.β.item():.4f}')
                if params.β_true is not None: ax3.axhline(params.β_true.item(), color='r', linestyle='--', label=f'True β = {params.β_true.item():.4f}')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('β')
                ax3.legend()
                ax3.grid(True)

        # Total Unweighted Loss
        if history.total_no_weight:
            ax6.semilogy(history.total_no_weight, 'g-', label='Total Loss')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Unweighted Loss')
            ax6.legend()
            ax6.grid(True)

        # Individual Losses plot
        if history.individuals: # ∇⋅J = ∇⋅Σ
            if len(history.individuals[0]) == 9:
                component_names = ["∇⋅J", "Jy No Flux", "∇⋅Σ (xy)", "∇⋅Σ (yy)", "ϕ Bulk", "ϕ BC", "u Sym.", "ϕ Sym.", "u Data"]
            else:
                component_names = ["∇⋅J", "Jy No Flux", "∇⋅Σ (xy)", "∇⋅Σ (yy)", "ϕ Bulk", "u Sym.", "ϕ Sym.", "u Data"]
            for i, indiv_loss in enumerate(zip(*history.individuals)): ax4.semilogy(indiv_loss, label=f'ℒ {component_names[i]}')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Weighted Losses')
            ax4.legend()
            ax4.grid(True)

        # Individual Unweighted Losses plot
        if history.individuals_no_weight: # ∇⋅J = ∇⋅Σ
            if len(history.individuals_no_weight[0]) == 9:
                component_names = ["∇⋅J", "Jy No Flux", "∇⋅Σ (xy)", "∇⋅Σ (yy)", "ϕ Bulk", "ϕ BC", "u Sym.", "ϕ Sym.", "u Data"]
            else:
                component_names = ["∇⋅J", "Jy No Flux", "∇⋅Σ (xy)", "∇⋅Σ (yy)", "ϕ Bulk", "u Sym.", "ϕ Sym.", "u Data"]
            for i, indiv_loss in enumerate(zip(*history.individuals_no_weight)): ax5.semilogy(indiv_loss, label=f'ℒ {component_names[i]}')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Unweighted Losses')
            ax5.legend()
            ax5.grid(True)

        # Highlight sharp changes in ϕ' across the domain
        findCFL(ax=ax2, trials=trials, params=params, n_points=grad_points, threshold=grad_threshold)
        ax2.legend()

        plt.tight_layout()
        plt.show(block=False)
        if epoch % config.SAVE_STEPS == 0: plt.savefig(paths.visu_dir / f'plot_epoch_{epoch}.png')
        plt.pause(0.15)
        plt.close(fig)
