## Three-body Problem
  - `LSTM-1st-dt-f-t-5-irr-basis-alone.pth` and `LSTM-2nd-dt-f-df-t-8-irr-basis-alone.pth` are the pre-trained 1st and 2nd order NDO model file. 
  - Codes for training above NDOs are in folder `trainingNDO`. 
  - Assign parameters in `run.sh` and run:
    ```bash
    bash run.sh
    ```
  - Adapted from https://github.com/juntang-zhuang/torch_ACA/ [1]

[1]  J. Zhuang, N. Dvornek, X. Li, S. Tatikonda, X. Papademetris, and J. Duncan. Adaptive checkpoint adjoint method for gradient estimation in neural ode. In International Conference on Machine Learning, pages 11639–11649. PMLR, 2020.