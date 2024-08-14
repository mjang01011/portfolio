# Activation Functions

The Universal Approximation Theorem states that a sufficiently large MLP with one hidden layer and non-linear activation functions can approximate any continuous function. However, this wouldn't be possible with just linear activation. Imagine stacking multiple linear layers. Each layer just performs a linear combination of the previous layer's outputs, so the entire network would essentially be equivalent to a single linear layer, limiting it to representing only linear relationships. Therefore, with non-linear activation, we can represent the complex non-linear relationship between the input and the output and represent a much more complicated function with deeper non-linear layers.

# Back Propagation

$$
% Notation
L = \text{number of layers} (L = 3) \\
n_i = \text{number of nodes in layer } i \\
a_i^j = \text{activation of the } j\text{-th node in layer } i \\
w_{ij}^k = \text{weight connecting the } k\text{-th node in layer } i-1 \text{ to the } j\text{-th node in layer } i \\
b_i^j = \text{bias term for the } j\text{-th node in layer } i \\
z_i^j = \text{weighted sum of inputs before activation in the } j\text{-th node of layer } i \\
\delta_i^j = \text{error term (delta) for the } j\text{-th node in layer } i \\
\eta = \text{learning rate} \\
y_i = \text{actual output of the network (for output layer, } L\text{)} \\
t_i = \text{target output value}

% Mean Squared Error (MSE)
E = \frac{1}{2} \sum_{j=1}^{n_L} (y_L^j - t_j)^2

% Error term (delta) for output layer (L)
\delta_L^j = (y_L^j - t_j) \cdot f'(z_L^j)

% Error term (delta) for hidden layers (1 to L-1)
\delta_i^j = f'(z_i^j) \cdot \sum_{k=1}^{n_{i+1}} \delta_{i+1}^k \cdot w_{i+1}^{kj} \quad \text{for } j = 1 \text{ to } n_i

% Update Weights and Biases
w_{ij}^k(\text{new}) = w_{ij}^k(\text{old}) - \eta \cdot \delta_i^j \cdot a_{i-1}^k \\
b_i^j(\text{new}) = b_i^j(\text{old}) - \eta \cdot \delta_i^j
$$





![image-20240523160257582](C:\Users\desti\AppData\Roaming\Typora\typora-user-images\image-20240523160257582.png)

# Logits

https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean

# Vanishing Gradient

We can't simply make our network deeper. The first issue is the vanishing gradient problem. 

# Batch Norm

https://www.youtube.com/watch?v=JJZZWM9tGp4

# Layer Norm

# Loss Landscape

# Dropout

# Regularization

