## Additional Experiment 1
We use `Init-NODE` to denote the NDO initialization version of NODE.

<img src="../spiral-initialization-loss.png" width="600">

---

## Additional Experiment 2
Animations of the training processes of `NDO-NODE`, `Init-NODE`, `Vanilla NODE` on the stiff ODE problem (Section 4.2), respectively. We use `Init-NODE` to denote the NDO initialization version of NODE.

Using NDO as an constrant (`NDO-NODE`) could continuously help the training process of NODE, while it may still meet neumerical problems when using NDO as the initialization of NODE (`Init-NODE`).

<img src="../stiff-comp.gif" width="800">

---

## Additional Experiment 3
We learn $X(t) = \frac{1}{t+0.01}$ by vanilla NODE and NDO-NODE, respectively.

<img src="../inverse-func.png" width="600">

We plot true derivation $\dot{X}(t) = -\frac{1}{(t+0.01)^2}$ and NDO estimations below.
<img src="grad-comp-inverse.png" width="600">

---
## Derivative Comparisons of NDO and Groundtruth

### 4.1.1 Planar Spiral Systems
<img src="grad-comp-spiral.png" width="600">

### 4.1.2 Damped Harmonic Oscillator
<img src="grad-comp-oscillator.png" width="600">

### 4.1.3 Three-body Problem
Derivatives for each body in 3D space.
<img src="grad-comp-threebody-total.png" width="800">

Derivatives for body 1 in each dimension.
<img src="grad-comp-threebody-1.png" width="800">

Derivatives for body 2 in each dimension.
<img src="grad-comp-threebody-2.png" width="800">

Derivatives for body 3 in each dimension.
<img src="grad-comp-threebody-3.png" width="800">

### 4.2 Stiff ODE
<img src="grad-comp-stiff.png" width="600">

### 4.3 Airplane Vibration Dataset
<img src="grad-comp-airplane.png" width="600">
