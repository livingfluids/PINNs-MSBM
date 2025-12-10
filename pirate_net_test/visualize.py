import matplotlib.pyplot as plt
import torch
import paths

def plot(epoch, trials, params):
    with torch.no_grad():
        y_plot = ((params.y_coll_ + 1.0) / 2.0 * params.H).cpu().numpy()
        y_plot_data_ = ((params.y_data_ + 1.0) / 2.0 * params.H).cpu().numpy()
        u_plot = (trials.u_trial(params.y_coll_) * params.u_max).cpu().numpy()
        ϕ_plot = trials.ϕ_trial(params.y_coll_).cpu().numpy()

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4, 3))

        # u plot
        axs[0].plot(y_plot, u_plot, label=f'u at epoch {epoch}')
        axs[0].plot(y_plot_data_, params.u_data_ * params.u_max, 'ko', markersize=3)
        axs[0].set_xlabel('y [m]')
        axs[0].set_ylabel('u [m/s]')
        axs[0].legend()
        axs[0].grid()

        # ϕ plot
        axs[1].plot(y_plot, ϕ_plot, label=f'ϕ at epoch {epoch}')
        axs[1].plot(y_plot_data_, params.ϕ_data, 'ko', markersize=3)
        axs[1].set_xlabel('ϕ [dimensionless]')
        axs[1].set_ylabel('u [m/s]')
        axs[1].legend()
        axs[1].grid()

        plt.tight_layout()
        plt.show(block=False)
        # if epoch % 50 == 0: plt.savefig(paths.visu_dir / f'plot_epoch_{epoch}.png')
        plt.pause(0.1)
        plt.close(fig)