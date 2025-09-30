# Learning hidden particle migration in concentrated particle suspension flow using Physics-Informed Neural Networks

## Some Results

Particle volume fraction prediction (red) compared to OpenFOAM data (black dots) for the inverse problem with known velocity data via a SA-PINN with self-adaptive weights, a Gauss expansion layer, and the finite difference method used for calculating derivatives within the PDE loss:
![SAPINN](assets/gauss_FDM_phi.png) 

## Scripts
	•	forward_FDM_Gauss_SA-PINN_phi_and_Ux.py - solves the forward problem for phi and Ux using a Gauss expansion layer and the FDM for calculating derivatives in the PDE loss
 	•	forward_Fourier_SA-PINN_phi_and_Ux.py - solves the forward problem for phi and Ux using a Fourier expansion layer
  	•	forward_Gauss_SA-PINN_phi_and_Ux.py - solves the forward problem for phi and Ux using a Gauss expansion layer
  
	•	inverse_FDM_Gauss_SA-PINN_phi_experimental.py - solves the inverse problem for phi for experimental data using a Gauss expansion layer and the FDM for calculating derivatives in the PDE loss
	•	inverse_Fourier_SA-PINN_phi_experimental.py - solves the inverse problem for phi for experimental data using a Fourier expansion layer
 	•	inverse_Gauss_SA-PINN_phi_experimental.py - solves the inverse problem for phi for experimental data using a Gauss expansion layer
  	•	inverse_PINN_Ux_experimental.py - solves the inverse problem for Ux for experimental data
 
	•	inverse_FDM_Gauss_SA-PINN_phi_and_beta_synthetic.py - solves the inverse problem for phi and beta for synthetic data using a Gauss expansion layer and the FDM for calculating derivatives in the PDE loss
 	•	inverse_FDM_Gauss_SA-PINN_phi_synthetic.py - solves the inverse problem for phi for synthetic data using a Gauss expansion layer and the FDM for calculating derivatives in the PDE loss
  	•	inverse_Fourier_SA-PINN_phi_and_beta_synthetic.py - solves the inverse problem for phi and beta for synthetic data using a Fourier expansion layer
   	•	inverse_Fourier_SA-PINN_phi_synthetic.py - solves the inverse problem for phi for synthetic data using a Fourier expansion layer
	•	inverse_Gauss_SA-PINN_phi_and_beta_synthetic.py - solves the inverse problem for phi and beta for synthetic data using a Gauss expansion layer
  	•	inverse_Gauss_SA-PINN_phi_synthetic.py - solves the inverse problem for phi for synthetic data using a Gauss expansion layer
   	•	inverse_PINN_Ux_synthetic.py - solves the inverse problem for Ux for synthetic data


The scripts are formatted similarly, where any differences have to do with the problem itself and are mentioned above. 

## Methodology

See [documentation.pdf](documentation.pdf) for a thorough review of the methodology in regards to PINN architecture and loss handling.

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

## How to Run

### Quick Start
The scripts automatically detect and use the best available compute device:
- **Apple Silicon Macs**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- **NVIDIA GPUs**: Uses CUDA for GPU acceleration  
- **CPU fallback**: Automatically falls back to CPU if no GPU is available

**Performance Note**: For some models/systems, CPU may actually be faster than GPU. All scripts include a `USE_GPU` flag to easily switch between GPU and CPU modes.

### Forward Problems (Simultaneous Training)
Run any of the forward problem scripts to train both Ux and ϕ simultaneously:

```bash
cd forward_problems
python forward_FDM_Gauss_SA-PINN_phi_and_Ux.py        # Gauss expansion + FDM
python forward_Fourier_SA-PINN_phi_and_Ux.py          # Fourier expansion
python forward_Gauss_SA-PINN_phi_and_Ux.py            # Gauss expansion
```

### Inverse Problems (Sequential Training)

#### For Synthetic Data:
1. **Train a Ux model first:**
   ```bash
   cd inverse_problems_synthetic
   python inverse_PINN_Ux_synthetic.py
   ```

2. **Then predict ϕ using your preferred method:**
   ```bash
   # Choose one of the following:
   python inverse_FDM_Gauss_SA-PINN_phi_synthetic.py     # Gauss + FDM (recommended)
   python inverse_Fourier_SA-PINN_phi_synthetic.py       # Fourier expansion
   python inverse_Gauss_SA-PINN_phi_synthetic.py         # Gauss expansion
   
   # For joint parameter estimation:
   python inverse_FDM_Gauss_SA-PINN_phi_and_beta_synthetic.py
   python inverse_Fourier_SA-PINN_phi_and_beta_synthetic.py
   python inverse_Gauss_SA-PINN_phi_and_beta_synthetic.py
   ```

#### For Experimental Data:
1. **Train a Ux model first:**
   ```bash
   cd inverse_problems_experimental
   python inverse_PINN_Ux_experimental.py
   ```

2. **Then predict ϕ using your preferred method:**
   ```bash
   # Choose one of the following:
   python inverse_FDM_Gauss_SA-PINN_phi_experimental.py  # Gauss + FDM (recommended)
   python inverse_Fourier_SA-PINN_phi_experimental.py    # Fourier expansion
   python inverse_Gauss_SA-PINN_phi_experimental.py      # Gauss expansion
   ```

### Configuration Options
Each script contains configurable parameters at the top:
- `USE_GPU`: Set to `False` to force CPU usage (useful if CPU is faster for your system)
- `data_file_1`: Choose between example datasets (True for example 1, False for example 2)
- `save_images`: Enable to save training progress images for animation
- `use_scheduler`: Enable learning rate scheduling
- Network architecture parameters (neurons, layers, learning rates, epochs)

**Device Selection Example:**
```python
USE_GPU = False  # Force CPU usage
# or
USE_GPU = True   # Use GPU if available (MPS on Mac, CUDA on NVIDIA)
```

### Output
- Trained models are saved in `saved_models/` directories
- Visualization images (if enabled) are saved in `saved_visuals/` directories
- Training progress and loss values are printed to console

## Troubleshooting

### NumPy Compatibility Issues
If you encounter errors like "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.1", this is due to PyTorch being compiled against NumPy 1.x while NumPy 2.0+ is installed.

**Solution:**
```bash
conda activate pinns  # or your environment name
conda install pytorch torchvision torchaudio "numpy<2" matplotlib pandas -c pytorch
```

### Environment Activation
Make sure you're in the correct conda environment before running scripts:
```bash
conda activate pinns  # or your environment name
```

### GPU vs CPU Performance
- **Apple Silicon Macs**: PyTorch automatically uses MPS (Metal Performance Shaders) for GPU acceleration
- **NVIDIA GPUs**: PyTorch uses CUDA if properly installed
- **CPU performance**: For Physics-Informed Neural Networks with smaller architectures, CPU may actually be faster than GPU due to overhead

**Benchmark your system**: Try both GPU and CPU modes to see which performs better:
```python
# In any script, change this line:
USE_GPU = False  # Force CPU
# vs
USE_GPU = True   # Use GPU if available
```

Check GPU availability:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A')"
```

## References

J. D. Toscano, V. Oommen, A. J. Varghese, Z. Zou, N. A. Daryakenari, C. Wu, and G. E. Karniadakis, “From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning,” 2024. [Online]. Available: Brown University, Division of Applied Mathematics.

K. L. Lim, R. Dutta, and M. Rotaru, “Physics informed neural network using finite difference method,” 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), IEEE, 2022, pp. 1828–1833.

A. D. Jagtap, D. Mitsotakis, and G. E. Karniadakis, “Deep learning of inverse water waves problems using multi-fidelity data: Application to Serre–Green–Naghdi equations,” Ocean Engineering, vol. 248, 2022, 110775.

Dbouk, Talib, Elisabeth Lemaire, Laurent Lobry, and Fady Moukalled. “Shear-induced particle migration: Predictions from experimental evaluation of the particle stress tensor.” Journal of Non-Newtonian Fluid Mechanics 198 (2013): 78–95. DOI: 10.1016/j.jnnfm.2013.03.006

McClenny, Levi D., and Ulisses M. Braga-Neto. “Self-adaptive physics-informed neural networks.” Journal of Computational Physics 474 (2023): 111722. DOI: 10.1016/j.jcp.2022.111722

M. Tancik, P. Srinivasan, B. Mildenhall, et al., “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains,” arXiv preprint arXiv:2006.10739, 2020.

Bilionis, Ilias¹; Hans, Atharva². A Hands‑on Introduction to Physics‑Informed Neural Networks. ¹ Mechanical Engineering, Purdue University, West Lafayette, IN; ² Design Engineering Lab, Purdue University, West Lafayette, IN.
