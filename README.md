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

## How to Run

	1.	Train a Ux model (synthetic or experimental)
	2.	Predict ϕ using your preferred script (Gauss expansion, Fourier expansion, etc.)

	or 

 	1. Ux and phi are trained simultaneously and within the same script for the forward problems

## References

J. D. Toscano, V. Oommen, A. J. Varghese, Z. Zou, N. A. Daryakenari, C. Wu, and G. E. Karniadakis, “From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning,” 2024. [Online]. Available: Brown University, Division of Applied Mathematics.

K. L. Lim, R. Dutta, and M. Rotaru, “Physics informed neural network using finite difference method,” 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), IEEE, 2022, pp. 1828–1833.

A. D. Jagtap, D. Mitsotakis, and G. E. Karniadakis, “Deep learning of inverse water waves problems using multi-fidelity data: Application to Serre–Green–Naghdi equations,” Ocean Engineering, vol. 248, 2022, 110775.

Dbouk, Talib, Elisabeth Lemaire, Laurent Lobry, and Fady Moukalled. “Shear-induced particle migration: Predictions from experimental evaluation of the particle stress tensor.” Journal of Non-Newtonian Fluid Mechanics 198 (2013): 78–95. DOI: 10.1016/j.jnnfm.2013.03.006

McClenny, Levi D., and Ulisses M. Braga-Neto. “Self-adaptive physics-informed neural networks.” Journal of Computational Physics 474 (2023): 111722. DOI: 10.1016/j.jcp.2022.111722

M. Tancik, P. Srinivasan, B. Mildenhall, et al., “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains,” arXiv preprint arXiv:2006.10739, 2020.

Bilionis, Ilias¹; Hans, Atharva². A Hands‑on Introduction to Physics‑Informed Neural Networks. ¹ Mechanical Engineering, Purdue University, West Lafayette, IN; ² Design Engineering Lab, Purdue University, West Lafayette, IN.
