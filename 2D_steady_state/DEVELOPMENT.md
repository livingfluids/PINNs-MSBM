
ssh mkd21@login.rcc.fsu.edu

Experiments:
- problem is there is a CFL on the inner wall 
- removing J dot n term results in non-zero phi values at the inner wall, but not the correct convergence
- removing phi neumann doesn't change the CFL
- using phi dirichlet instead of neumann still causes CFL
- removing sigmoid from phi trial function seemed to help, but still there is a CFL-like region


model: 

the goal is to build a PINN to solve the bifurcation suspension balance problem and validate it using data from an OpenFOAM solution that uses the same PDEs and BCs. currently, we are testing the model without the lift force included in the migration term. 

The PINN is solving the inverse problem, with data enforced at the inlet. additional data is used to act as a benchmark to validate the model with the MSE and profile visualizations, but are not intended to be included in the loss. 

the OpenFOAM solution shows a peak throughout the centerline of the parent branch, which then splits at the bifurcation's V corner. the new peaks in the daughter branches are not at the centerline, but at the walls which formed the V corner. aditionally, the particles seem to experience a buildup at the V corner. for reference, the max phi value at the V corner from the daughter branch data profile is around 0.48, while the max phi value for the inlet at the centerline peak (where the data is actually enforced) is around 0.34.

the data profiles are enforced/compared to at their exact positions according to the OpenFOAM model

problem: 

this model performs performs well overall. it is very obvious that the model is able to predict the general shape of the solution. however, the regions where large phi values are expected seem to be underpredicted at convergence. at the parent data benchmark profile, the centerline peak is slightly smaller than expectd by around 0.05. at the daughter benchmark profile, the wall peak also smaller than expected by around 0.1-0.15. otherwise, away from these regions the phi solution matches almost perfectly. 

the model's performance depends on the NN architecture, optimizer parameters, collocation point count and positions, loss scheme parameters and design, etc., so it is difficult to identify any singular cause of the problem. however, the tests i have done so far seem to produce similar results with the same phi-underprediction problem.

tests:

I may be forgetting some of the tests i have done, but the ones i remember are the following
- number of hidden layers (3 vs 5)
- loss scheme (maxGradMagnitude vs MeanAbsGrad only)
- various GRAD_NORM_EPOCH_INTERVAL, SCALE, NEURONS, ACTIVATION, N_PTS_PROPOSAL, N_PTS_BDR configurations
- sign swapping within the x and y momentum terms
- the values in parameters.yaml are all correct to the OpenFOAM model, but i have tried larger phi_max values
- zero Reynolds number

general notes:

regarding the underpredicted peaks, it is almost as if the particles are diffusing too quickly and not compacting enough.

this might be a scaling/normalization or loss term choice error

some loss terms i know ere not included in the OpenFOAM model, like the flux and ratio terms, however removing them did not seem to make any difference

the solution, depending on the parameter configurations, will frequenty predict large, sharp particle buildups at all walls, which isn't present on the OpenFOAM data. 
