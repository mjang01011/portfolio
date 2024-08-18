# Neural Networks

Neural networks are composed of neurons connected by edges. Each neuron receives inputs, and each edge has a weight that multiplies the input value. This value is then added to a bias term and passed through an activation function. The weights determine the importance of each neuron's input. The goal of a neural network is to learn the optimal set of weights and biases so that the model returns the desired output for a given input.

A neural network is organized into three main layers: input layer, hidden layers, and output layer.

![Artificial Neural Networks and its Applications - GeeksforGeeks](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20230602113310/Neural-Networks-Architecture.png)

- **Input Layer:** The first layer where the network receives data inputs.
- **Hidden Layers:** Intermediate layers where neurons process inputs using weights, biases, and activation functions to extract and transform features. These layers are crucial for processing information from the input layer and passing it to the output layer to produce the desired outputs.
- **Output Layer:** The final layer where the network produces the output based on the processed data from the hidden layers.

When a network has many hidden layers, it is called a Deep Neural Network (DNN). When each neuron from one layer is connected to every neuron in the next layer, the layer is described as fully connected (FC). If all the layers are fully connected, the network is referred to as a multilayer perceptron (MLP).

Overall, a neural network functions to link the input to the output. The objective is to find a function that returns the desired output from a given input by learning the set of weights and biases.

How can we effectively set up this learning process? We can start simply by considering a linear regression problem. Given a set of data points, we want to find a linear function \( y = ax + b \) with the values for the parameters \( a \) and \( b \) that minimize the difference between the actual output \( y \) and the predicted output ŷ. This difference is defined as the loss, or the cost. It is important to define the loss based on the problem we want to solve. Here, we can use mean squared error (MSE) as our loss function. We can then determine \( a \) and \( b \) that minimize the loss by plotting the loss function for different \( a \) and \( b \) values.

But we soon realize that this is not the most efficient way to find the optimal values for \( a \) and \( b \) to minimize the loss function. As the number of parameters increases, the search space grows exponentially, making it impractical to find the optimal values by brute force. A more mathematical and systematic approach to minimize the loss is called gradient descent.

## Gradient Descent

We start by randomly initializing the values of \( a \) and \( b \). Then, we iteratively adjust these values in the direction that reduces the loss. How do we determine this direction? From calculus, we know that the gradient of a function at a specific point represents the direction of the steepest ascent. To minimize the loss function, we move in the opposite direction of the gradient. Therefore, we subtract the gradient of the loss function from our current values of \( a \) and \( b \) to move from our current position on the loss function towards the direction that minimizes the loss.

However, simply subtracting the gradient from our current position doesn't immediately guide us to the optimal location. We must perform the gradient descent process iteratively to approach the optimal solution. During this process, we may encounter challenges such as overshooting the minimum, making insufficient progress, or oscillating back and forth on the loss function.

## Learning Rate

We introduce a hyperparameter known as the learning rate (LR), often denoted as alpha. This constant scales the gradient of the loss function, determining the magnitude of each step we take towards minimizing the loss. Adjusting the learning rate allows us to control the rate of convergence towards the minimum of the loss function.

## Issues

There are subtleties that demand close consideration. Firstly, despite our careful selection of a suitable learning rate, there remains a risk of it being either too large or too small, potentially leading to overshooting, oscillation, or confinement to a local minimum rather than the global minimum. Secondly, the random initialization of 𝑎 and 𝑏 values presents its own set of challenges. It turns out that initial values of the weights can have a significant impact on model training. Lastly, there's the issue of computational efficiency when calculating the gradient. In practical implementation for neural networks, computing the gradient direction with all the input data may require significant time and resources.

## Weight initialization

The first concern can be addressed through a technique called learning rate scheduling. By controlling the learning rate during model training, we can, for example, decrease it as gradient descent iterations progress, thus preventing overshooting. As for the second issue, employing smart strategies for weight initialization can effectively tackle it. There are three well known weight initialized techniques, each proposed by LeCun, Xavier, and Kaiming.

First of all, it is important to note that initializing the weights to 0 wouldn't allow the model to learn anything. Similarly, extremely big or small weight initialization can lead to exploding or vanishing gradients.

## 1. LeCun Initialization

**Method:** LeCun initialization is designed for activation functions like sigmoid or hyperbolic tangent (tanh). It initializes weights using a distribution with zero mean and a variance of 1/n where n is the number of input units in the layer.

**Formula:**
$$
W \sim \mathcal{N}\left(0, \frac{1}{n}\right)
$$
For a uniform distribution:
$$
W \sim \text{Uniform}\left(-\sqrt{\frac{3}{n_{\text{in}}}}, \sqrt{\frac{3}{n_{\text{in}}}}\right)
$$
**Purpose:** This method helps to keep the variance of activations and gradients roughly the same across layers by adjusting the variance based on the input dimensions, which mitigates the vanishing or exploding gradient problem. This helps mitigate the vanishing gradient issue for sigmoid or tanh functions, but may not be useful for activations like ReLU.

## 2. Xavier Initialization

**Method:** Xavier initialization, also known as Glorot initialization, aims to maintain the variance of activations and gradients across layers by considering both the input and output dimensions. It works well with activation functions like tanh or sigmoid. Weights are initialized from a distribution with zero mean and variance
$$
\frac{2}{n_{\text{in}} + n_{\text{out}}}
$$
, where 
$$
n_{\text{in}}
$$
 is the number of input units and
$$
n_{\text{out}}
$$
is the number of output units.

**Formula:**
$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
$$
Alternatively, for a uniform distribution:
$$
W \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)
$$
**Purpose:** Balances the variance of activations and gradients, reducing the risk of vanishing or exploding gradients in deep networks. However, with ReLU activation, it may also be hard to resolve the vanishing gradient issue since Xavier initialization uses smaller variance compared to Kaiming initialization. With smaller weights and ReLU, in the initial phase nodes can output 0 which can lead to poor training.

## 3. Kaiming (He) Initialization

**Method:** Kaiming initialization, also known as He initialization, is particularly suited for activation functions like ReLU (Rectified Linear Unit). It initializes weights with a variance of
$$
\frac{2}{n_{\text{in}}}
$$
, where 
$$
n_{\text{in}}
$$
is the number of input units.

**Formula:**
$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
$$
For ReLU activation, a scaled variant is used:
$$
W \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right)
$$
**Purpose:** Addresses the problem of gradients becoming too small or too large by considering the non-linearity introduced by ReLU activations. Unlike the other two initializations, the large variance can lead to saturation when using sigmoid or tanh activations, worsening the vanishing gradient issue.

In summary:

- **LeCun Initialization:** Suitable for tanh or sigmoid activations; scales weights to maintain variance.
- **Xavier Initialization:** Balances variance of activations for tanh or sigmoid; uses an average of input and output units.
- **Kaiming Initialization:** Optimized for ReLU activations; scales weights based on the number of input units to maintain variance.

## Stochastic & Batch Gradient Descent

If computing the gradient of the loss function with all the data poses computational challenges, a straightforward and intuitive solution is to randomly select one data point at a time to calculate the gradient. This approach is known as stochastic gradient descent (SGD). In essence, we perform gradient descent for each individual data point and refrain from updating the gradient for that data point until we have completed SGD for all other available data. Once we have traversed through all the data using SGD, we call this a single epoch and for the next epoch we reconsider all the data and repeat the SGD process until reaching the desired loss. In other words, three epochs mean that we have considered the entire data three times.

It's worth noting that while the gradient always indicates the direction of the steepest ascent, using a single data point to compute the gradient does not necessarily align with the steepest ascent direction on the original loss function. This discrepancy arises because the loss function is derived from all the data. However, employing SGD guides the weights towards minimizing the loss for that specific data point only. Nonetheless, this iterative approach still facilitates progress towards the local minimum of the loss function, as each individual data point is iteratively considered. Moreover, SGD offers a potential means to avoid being trapped in a local minimum and to converge towards a better local or global minimum.

While stochastic gradient descent (SGD) is efficient, it can still lead to excessive oscillation along the loss function towards the minimum when considering only one data point at a time. Hence, mini-batch SGD is often preferred, where a batch of data is considered for each gradient descent iteration. As a side note, let's say we have N data available. When we are implementing mini-batch SGD of size k, what if N is not divisible by k? Do we not consider the remaining N % k data for our last iteration through the data? The answer is we would use the remaining < k data for the last iteration, since we want to determine the direction that minimizes the loss for all our data. In practice, 8k is considered as the threshold for the batch size (under the experimental condition where the learning rate is scaled accordingly to the scaled increase in batch size and learning rate warmup). It is speculated that the larger batch size we use, the more likely we will fall into a local minimum.

## References

Goyal, P., Dollár, P., Girshick, R. B., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* CoRR, abs/1706.02677