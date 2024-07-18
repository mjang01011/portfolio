## Momentum

Mini-batch SGD can oscillate as we perform gradient descent on the loss function. To resolve this, we consider the previous gradient directions to add 'momentum' to the descent. Imagine that the mini-batch SGD oscillates as shown below.

![image-20240522144312411](C:\Users\desti\AppData\Roaming\Typora\typora-user-images\image-20240522144312411.png)

By considering the previous gradient directions, the left and right oscillations cancel out, and the forward momentum towards the minima accumulates, resulting in a faster gradient descent compared to the mini-batch SGD. One point to consider is that near the minima, the gradient 'rotates' around the minima.

## RMSProp

The intuitive explanation behind Root Mean Square Propagation (RMSProp) is that we are essentially adjusting different learning rates to different parameters. This is done by considering the magnitude of each parameter's gradient. Parameters with large gradients will have their learning rate reduced, whereas parameters with small gradients will have their learning rate increased. It also considers previous gradients by conducting a running average of magnitudes of squares of those gradients.

## Adam

Adaptive Moment Estimation (Adam)



https://ruder.io/optimizing-gradient-descent/