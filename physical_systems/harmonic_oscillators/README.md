## Damped Harmonic Oscillators
  - `N10k-1st-100-Osin50xd50-no1x-irr-10pn-LSTM-t-f-dt-0006.pth` and `N10k-2th-100-Osin50xd50-no1x-irr-10pn-LSTM-t-f-df-dt-0813.pth` are the pre-trained 1st and 2nd order NDO model file.
  - Assign parameters in `run.sh` and run:
    ```bash
    bash run.sh
    ```

  - Adapted from https://github.com/a-norcliffe/sonode/tree/master/experiments/damped_oscillators [1]

[1] A. Norcliffe, C. Bodnar, B. Day, N. Simidjievski, and P. Liò. On Second Order Behaviour in
350 Augmented Neural ODEs. arXiv:2006.07220 [cs, stat], Oct. 2020.

--- 
This contains the damped oscillators experiment. 30 harmonic oscillators, with the same ode, but 30 random pairs
of initial positions and velocities. 