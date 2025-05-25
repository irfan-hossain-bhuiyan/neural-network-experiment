### 1. **Gradient Descent**
Before we get into momentum or Adam, let's start with the basics.

In neural networks, we aim to minimize a loss function during training. This process is often referred to as **optimization**. The 
most common optimization algorithm used today is **gradient descent**, which updates model parameters in the direction of the 
steepest decrease of the loss function.

The mathematical formula for gradient descent is:
$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t)
$$
Where:
-  \theta_t \) is the parameter at step \( t \).
- \( \eta \) is the learning rate.
- \( J(\theta_t) \) is the loss function.

---

### 2. **Momentum**
Momentum is an optimization technique that helps accelerate gradient descent by adding a fraction of the previous update to the 
current one. This helps in speeding up convergence and reducing oscillations, especially during training when the gradients might 
be noisy or pointing in slightly incorrect directions.

The mathematical formula for momentum is:

$$
v = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t
$$
$$
\theta_t = \theta_{t-1} - \eta \cdot v_t
$$

Where:
- \( v \) is the velocity or momentum term.
- \( \beta \) is a hyperparameter that controls how much of the previous velocity to retain (commonly set to 0.9).
- \( g_t = \nabla_\theta J(\theta_{t-1}) \) is the gradient at step \( t \).
- \( \eta \) is the learning rate.

The momentum term \( v_t \) accumulates the gradients over time, which helps in building up force in the direction of the true 
minimum. This smooths out the parameter updates and can lead to faster convergence.

---

### 3. **Adam (Adaptive Moment Estimation)**
Adam is an optimization algorithm that combines the advantages of both AdaGrad and RMSProp. It computes adaptive learning rates for 
each parameter by maintaining estimates of the first and second moments of the gradients.

The mathematical formulas for Adam are:

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$
$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

Where:
- \( m_t \) is the first moment estimate (mean of gradients).
- \( v_t \) is the second moment estimate (variance of gradients).
- \( \beta_1 \) and \( \beta_2 \) are hyperparameters that control the decay rates of the moment estimates (commonly set to 0.9 and 
0.99, respectively).

To get the bias-corrected estimates:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$
$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Finally, the parameter update is:

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Where:
- \( \epsilon \) is a small constant (like 1e-8) to prevent division by zero.

---

### Summary of Key Differences
- **Momentum** accumulates the velocity of gradients over time, which helps in speeding up convergence and reducing oscillations.
- **Adam** adapts learning rates for each parameter by combining the first and second moments of the gradients. It is generally 
considered a "one-pass" algorithm because it automatically adjusts the learning rate per parameter.

Both momentum and Adam are widely used in practice because they improve the training efficiency of neural networks, especially when 
dealing with large datasets and complex models.

---

If you'd like, I can provide an example or even a simple code snippet to illustrate how these algorithms work! Let me know.
