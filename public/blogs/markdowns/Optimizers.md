# Optimizers

Optimizers are algorithms or methods used to change the attributes of your neural network such as weights to minimize the loss function. This blog will examine different optimizers often used in deep learning.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">    <div>     <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhchldRUkLsKUzf6uuEx_zrYRjoIdizj059vRX25LxGQGp2xzoRWYC1-6GxvHyGF9uK2v4j5HI2YQTCrC4DXp9JnRnWY40NYwZjuUVXVV2NWpuzoR1LbPgU3a0UQEfWH4KqIS9i/s400/Long+Valley+-+Imgur.gif" alt="Image 1" style="width:100%;">   </div>    <div>     <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi9JzBXNfW9sm4yKCyV_MhyfOJiwqwocsPOpnY8i_pLy0AcN0rcz9LTn6Wp3-UG4CQYM_vU1CxKfqdKdS9KkZmc0AWu1gJxiCfYfd3DZRpz5Btx8q25Qs9I9PJFnEpxsHMUNdqH/s400/Saddle+Point+-+Imgur.gif" alt="Image 2" style="width:100%;">   </div>    <div>     <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjUvhiRVEra6dZlJFFoR-lYbtVpFCmM41IYACOLukmWUHSnsqABa517IuKtYLHMNXj75wTQtHbr_AlGxYVItrUA3QHBvecLUjpxdyGAiJxXO0uMMV9bD7NW54ePr9OH_JJx3x5I/s400/s25RsOr+-+Imgur.gif" alt="Image 3" style="width:100%;">   </div>    <div>     <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjLZPaZUGrzT1_HWfcLnyx-3idRZyw8c3WAvrlyMX3nhLpTDnkoezzDWeiOqBHlItng5Umyx5Hm6iOhpPB6eKxtotr0aH_k2GBB9PAAePMRpUu1cJTEFcdeBpoTj2B0Eb1h84dx/s400/Beale&%23039;s+function+-+Imgur.gif" alt="Image 4" style="width:100%;">   </div>  </div>

Top Left: **Long valley:** Algos without scaling based on gradient information really struggle to break symmetry here - SGD gets no where and Nesterov Accelerated Gradient / Momentum exhibits oscillations until they build up velocity in the optimization direction. Algos that scale step size based on the gradient quickly break symmetry and begin descent.

Top Right: **Saddle point:** Behavior around a saddle point. NAG/Momentum again like to explore around, almost taking a different path. Adadelta/Adagrad/RMSProp proceed like accelerated SGD.

Bottom Left: **Noisy moons:** This is logistic regression on noisy moons dataset from sklearn which shows the smoothing effects of momentum based techniques (which also results in over shooting and correction). The error surface is visualized as an average over the whole dataset empirically, but the trajectories show the dynamics of minibatches on noisy data. The bottom chart is an accuracy plot.

Bottom Right: **Beale's function:** Due to the large initial gradient, velocity based techniques shoot off and bounce around - adagrad almost goes unstable for the same reason. Algos that scale gradients/step sizes like adadelta and RMSProp proceed more like accelerated SGD and handle large gradients with more stability.

Image & Description Credit: [Alec Radford's animations for optimization algorithms](https://www.denizyuret.com/2015/03/alec-radfords-animations-for.html)

## Gradient Descent

$$
\theta := \theta - \eta \nabla_\theta J(\theta)
$$

- θ: Model parameters (weights and biases).
- η: Learning rate, a scalar that determines the step size.
- ∇θ J(θ): Gradient of the cost function with respect to parameters.

Gradient descent is the most basic type of optimization algorithm. As shown above, it is a first-order optimization algorithm that depends on the first order derivative of the loss function. This method is very intuitive and easy to understand, but may get stuck at a local minima and requires large memory as it calculates gradient on the whole dataset.

## Stochastic Gradient Descent (SGD)

$$
\theta := \theta - \eta \nabla_\theta J(\theta; x^{(i)}, y^{(i)})
$$

- ∇θ J(θ; x^{(i)}, y^{(i)}): Gradient of the cost function with respect to a single training example.

SGD is similar to that of vanilla gradient descent, except now we update the model parameters after computing the loss on each training example. For instance, if we have 100 training data, SGD will update the model parameters 100 times instead of making a single update after computing the loss over the entire training data like gradient descent. Now, we can use less memory and perhaps move away from a local minima as the variance is large since it only considers a single training example at a time. This also serves as a downside of SGD, as the path towards the minima can be very noisy and take longer time to achieve convergence.

## Mini Batch Gradient Descent

$$
\theta := \theta - \eta \nabla_\theta J(\theta; x^{(i:i+n)}, y^{(i:i+n)})

$$

Instead of calculating the loss over a single training example to update the parameters, mini batch SGD updates the parameters after every batch with n samples. Now, the optimizer has less variance and noise as it progresses towards the minima.

## Momentum

$$
v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta) \\
\theta := \theta - v_t
$$

- v_t: Velocity or moving average of gradients at time step *t*.
- γ: Momentum term, controls the influence of previous gradients (typically 0.9).

Mini batch SGD can still oscillate as we perform gradient descent on the loss function. To resolve this, we consider the previous gradient directions to add 'momentum' to the descent. Imagine that the mini batch SGD oscillates as shown in the bottom left figure.

![image-20240522144312411](https://github.com/mjang01011/portfolio/blob/main/public/blogs/markdowns/images/image-20240522144312411.png?raw=true)

The right figure shows the same process but with momentum. By considering the previous gradient directions, the left and right oscillations cancel out, and the forward momentum towards the minima accumulates, resulting in a faster gradient descent compared to the mini batch SGD. One point to consider is that near the minima, the gradient 'rotates' around the minima. Typically, 0.9 is a good starting point for the momentum.

## Nesterov Accelerated Gradient (NAG)

$$
\tilde{\theta}_t = \theta_t - \gamma v_{t-1} \\
v_t = \gamma v_{t-1} + \nabla_\theta J(\tilde{\theta}_t) \\
\theta_{t+1} = \theta_t - \eta v_t
$$

![img](https://blog.kakaocdn.net/dn/yB5qn/btrMHJL8INF/7O2idSodKUySlmAyztWsSk/img.jpg)

NAG is similar to that of momentum, but differs in the aspect that NAG first makes a "lookahead" gradient step in the direction of the momentum and then computes the gradient at this lookahead position to adjust the update. This anticipates the future position of the parameters to prevent overshooting from large momentum.

## Root Mean Square Propagation (RMSProp)

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) (\nabla_\theta J(\theta))^2 \\\\
\theta := \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla_\theta J(\theta)
$$

- E[g^2]_t: Exponential moving average of squared gradients at time step *t*.
- γ: Decay rate, a value close to 1 (typically 0.9).
- η: Learning rate.
- ε: A small constant added for nonzero division and stability (e.g., 1e-8).

The intuitive explanation behind RMSProp is that we are essentially adapting learning rates individually to different parameters. This is done by considering the magnitude of each parameter's gradient. Parameters with large gradients will have their learning rate reduced, whereas parameters with small gradients will have their learning rate increased. It also considers previous gradients by conducting a running average of magnitudes of squares of those gradients. It smooths out the oscillations that can occur due to large gradients and helps in speeding up convergence.

## Adaptive Gradient (Adagrad)

$$
G_t = G_{t-1} + (\nabla_\theta J(\theta))^2 \\\\
\theta := \theta - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_\theta J(\theta)
$$

- G_t: Accumulates the square of the gradients at each time step.

Adagrad is also an adaptive learning rate algorithm like RMSProp, but differs in the way it adapts the learning rate. Adagrad uses cumulative sum of squared gradients, whereas RMSProp uses a running average of squared gradients. The biggest issue with Adagrad is that the learning rate vanishes as we continue the training, as the accumulation can grow very large over time.

## AdaDelta

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) (\nabla_\theta J(\theta))^2 \\\\
\theta := \theta - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \nabla_\theta J(\theta)
$$

AdaDelta aims to reduce the monotonically decreasing learning rate of Adagrad. There are two main modifications: 1) it uses exponetially decaying average of squared gradients instead of their cumulative sum which allows the optimizer to adapt to more recent gradients, 2) it uses a decaying constant (same as that of RMSProp).

## Adaptive Moment Estimation (Adam)

Adam optimizer is basically the combination of both momentum and RMSProp. 

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \\\\
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\\\
\theta := \theta - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

- Exponential Decay Rates (β1,β2): Control the decay rates for the moving averages of the gradient and its square.

![Everything you need to know about Adam Optimizer | by Nishant Nikhil |  Medium](https://miro.medium.com/v2/resize:fit:1100/1*zfdW5zAyQxge85gA_mFPYg.png)

Algorithm source: **Adam: A Method for Stochastic Optimization**, Diederik P. Kingma and Jimmy Ba, 2017 [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

The above algorithm may look complicated at first, but we come to realize that it is literally the combination of both momentum and RMSProp. The numerator (m) represents that of momentum and the denominator (sqrt of v) represents that of RMSProp.

## Summary

The bottom chart summarizes the progression from gradient descent to Adam optimizer.

![image](https://github.com/mjang01011/portfolio/blob/main/public/blogs/markdowns/images/image-optimizer.png?raw=true)

Ruder, Sebastian. "An overview of gradient descent optimization algorithms." [arXiv:1609.04747](https://arxiv.org/abs/1609.04747) (2016) is another great source for understanding various optimizers. It also talks about AdaMax, Nadam, parallelizing SGD, and various strategies for optimizing SGD.