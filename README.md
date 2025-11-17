# Learning hidden particle migration in concentrated particle suspension flow using Physics-Informed Neural Networks

This repository contains a unified Physics-Informed Neural Network (PINN) framework for learning particle migration in concentrated suspensions.
It supports forward and inverse problems, synthetic and experimental data, single-PINN or two-PINN architectures, Fourier/Gauss mappings, and optional finite-difference PDE residuals.

All functionality is driven by a single configuration file (config.py) and a single driver script (main.py).

## Overview

This PINN framework predicts one, two, or all of the following:
- The particle volume fraction ϕ(y)
- The suspension velocity profile u(y)
- The lift force coefficient β

### Steady State Problem

Inside `steady_state_problem/`:
- PINN related configurations---dataset selection, forward vs inverse mode, mapping layers, PINN architecture, training procedure, saving, visualization, and loss-handling---are controlled from `config.py`.
- All models are run from `main.py`, but you will never modify this file. 

## Methodology

See (link to paper)

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- GPU support (optional but recommended for faster training):
  - **Apple Silicon Macs**: MPS (Metal Performance Shaders) - automatically detected
  - **NVIDIA GPUs**: CUDA support
  - **CPU**: Works on all systems (may be faster for smaller models)

### Option 1: Using pip (Python package manager)

Install the required packages using pip:
```bash
pip install torch torchvision matplotlib numpy pandas pathlib
```

For GPU support (recommended), install PyTorch with CUDA:
```bash
# For CUDA 11.8 (check your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Option 2: Using conda (Recommended for better dependency management)

**First, install Miniconda/Anaconda if you don't have it:**

**On macOS:**
```bash
# Check your architecture first
uname -m

# For Apple Silicon Macs (M1/M2/M3/M4) - if output is 'arm64'
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# For Intel Macs (x86_64) - if output is 'x86_64'
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Follow the installer prompts, then restart your terminal
```

**On Linux:**
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the installer prompts, then restart your terminal
```

**On Windows:**
- Download the Miniconda installer from: https://docs.conda.io/en/latest/miniconda.html
- Run the `.exe` file and follow the installation wizard

**Then create and activate the conda environment:**
```bash
conda create -n pinns python=3.9
conda activate pinns
```

**Install PyTorch based on your system:**

**For macOS (Apple Silicon - M1/M2/M3/M4):**
```bash
# PyTorch with MPS (Metal Performance Shaders) support for GPU acceleration
conda install pytorch torchvision torchaudio -c pytorch
conda install matplotlib "numpy<2" pandas
```

**For Linux/Windows with NVIDIA GPU (CUDA support):**
```bash
# Check your CUDA version first: nvidia-smi
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib "numpy<2" pandas
```

**For CPU-only (any platform):**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib "numpy<2" pandas
```

## References

J. D. Toscano, V. Oommen, A. J. Varghese, Z. Zou, N. A. Daryakenari, C. Wu, and G. E. Karniadakis, “From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning,” 2024. [Online]. Available: Brown University, Division of Applied Mathematics.

K. L. Lim, R. Dutta, and M. Rotaru, “Physics informed neural network using finite difference method,” 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), IEEE, 2022, pp. 1828–1833.

A. D. Jagtap, D. Mitsotakis, and G. E. Karniadakis, “Deep learning of inverse water waves problems using multi-fidelity data: Application to Serre–Green–Naghdi equations,” Ocean Engineering, vol. 248, 2022, 110775.

Dbouk, Talib, Elisabeth Lemaire, Laurent Lobry, and Fady Moukalled. “Shear-induced particle migration: Predictions from experimental evaluation of the particle stress tensor.” Journal of Non-Newtonian Fluid Mechanics 198 (2013): 78–95. DOI: 10.1016/j.jnnfm.2013.03.006

McClenny, Levi D., and Ulisses M. Braga-Neto. “Self-adaptive physics-informed neural networks.” Journal of Computational Physics 474 (2023): 111722. DOI: 10.1016/j.jcp.2022.111722

M. Tancik, P. Srinivasan, B. Mildenhall, et al., “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains,” arXiv preprint arXiv:2006.10739, 2020.

Bilionis, Ilias¹; Hans, Atharva². A Hands‑on Introduction to Physics‑Informed Neural Networks. ¹ Mechanical Engineering, Purdue University, West Lafayette, IN; ² Design Engineering Lab, Purdue University, West Lafayette, IN.
