## Stiff ODE
  - `N10k-1st-100-Osin50xd50-no1x-irr-10pn-LSTM-t-f-dt-0006.pth` is the pre-trained 1st order NDO model file.
  - Adapted from https://github.com/arnabgho/steer [1]. Thus please follow the instructions on [STEER](https://github.com/arnabgho/steer) to install the requirements before running following codes. 
  - Assign parameters in `run.sh` and run following codes for stiff ODE in Figure 4.
    ```bash
    bash run.sh
    ```
  - Assign parameters in `run_sin.sh` and run following codes for stiff ODE with sin term.
    ```bash
    bash run_sin.sh
    ```
  - For results of STEER, please refer to [STEER](https://github.com/arnabgho/steer).

[1] A. Ghosh, H. Behl, E. Dupont, P. Torr, and V. Namboodiri. Steer: Simple temporal regularization for neural ode. Advances in Neural Information Processing Systems, 33, 2020.