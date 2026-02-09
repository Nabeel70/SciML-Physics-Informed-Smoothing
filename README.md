# Scientific Machine Learning: Physics-Informed Neural Smoothing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Scientific-red)](https://pytorch.org/)
[![SciML](https://img.shields.io/badge/Domain-SciML-green)]()

## 📜 Abstract
This project implements a **Physics-Informed Neural Network (PINN)** approach to solve the problem of overfitting in noisy experimental data. By designing a custom loss function that incorporates **Second-Order Derivative Regularization** (Sobolev smoothing), the model learns the underlying physical law ($y=\sin(x)$) rather than fitting the Gaussian noise.

This approach demonstrates the core principle of **Scientific Machine Learning**: utilizing inductive biases (smoothness constraints) to govern neural network convergence.

## 🧮 Mathematical Formulation

Standard neural networks minimize the Mean Squared Error (MSE), which often leads to overfitting in high-variance regimes:

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

To enforce physical plausibility, we introduce a **Smoothness Regularizer** based on the curvature of the function. Using PyTorch's **Automatic Differentiation (AutoGrad)** engine, we compute the exact second derivative of the network output with respect to the input:

$$\mathcal{L}_{smooth} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{\partial^2 \hat{y}}{\partial x^2} \right)^2$$

The final optimization objective becomes:

$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda \cdot \mathcal{L}_{smooth}$$

Where $\lambda$ is a hyperparameter balancing data fidelity and physical constraints.

## 🛠️ Implementation Details
- **Architecture:** Fully Connected Neural Network (FCN) with Tanh activations (differentiable $\mathcal{C}^\infty$).
- **Differentiation:** `torch.autograd.grad` is used to compute higher-order derivatives directly from the computation graph.
- **Optimization:** Gradient Descent via Adam Optimizer.

## 📊 Results
The ablation study demonstrates that:
1.  **$\lambda = 0$ (Standard MSE):** The model captures high-frequency noise (Overfitting).
2.  **$\lambda = 0.1$ (Reinforced Smoothing):** The model recovers the ground truth function $\sin(x)$ and ignores Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.2)$.

*(See notebook for visualization plots)*

## 🚀 Usage

```bash
# Clone the repository
git clone [https://github.com/Nabeel70/SciML-Physics-Informed-Smoothing](https://github.com/Nabeel70/SciML-Physics-Informed-Smoothing)

# Install dependencies
pip install torch numpy matplotlib jupyter

# Run the experiment
jupyter notebook reinforced_smoothing.ipynb