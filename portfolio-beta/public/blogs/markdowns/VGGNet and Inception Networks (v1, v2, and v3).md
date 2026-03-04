# VGGNet & Inception Net

## VGGNet

Original Paper: **Simonyan, Karen, and Andrew Zisserman.** "Very Deep Convolutional Networks for Large-Scale Image Recognition." *arXiv preprint arXiv:1409.1556* (2015). Available at: https://arxiv.org/abs/1409.1556

The key goal of VGGNet was to determine if deeper network leads to better performance.

![VGG-Net Architecture Explained. The company Visual Geometry Group… | by  Siddhesh Bangar | Medium](https://miro.medium.com/v2/resize:fit:1400/1*VPm-hHOM14OisbFUU4cL6Q.png)

- **Architecture:** Consists of CNN layers using 3x3 convolution filters and 3 FC layers.
- VGGNet showed that multiple 3x3 convolutions perform equally compared to a single 7x7 convolution. The reason why multiple, smaller convolutions are preferred is that it adds more nonlinearity which allows the model to identify key features and reduces the number of weights and computations. 7x7 convolution needs 49 weights whereas three 3x3 convolutions require 27 weights.

**Pros:**

- Simplicity and ease of understanding.
- Good performance on various image recognition tasks.

**Cons:**

- Computationally expensive and requires a lot of memory.

  

## Inception V1 (GoogLeNet)

Original Paper: **Szegedy, Christian, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich.** "Going Deeper with Convolutions." *arXiv preprint arXiv:1409.4842* (2014). Available at: https://arxiv.org/abs/1409.4842

The easy way of boosting the model performance is to make the model deeper and wider. However, this poses three problems: 1) there are more weight parameters to learn, 2) it is computationally very expensive, 3) it can lead to overfitting. Inception V1 tries to solve this issue with the inception module. Inception V1 has 12 times fewer parameters than AlexNet, but performs much better. 

![Google Inception Model.](https://norman3.github.io/papers/images/google_inception/f01.png)

- **Inception Module:** Uses multiple convolutional and pooling operations at different scales, concatenated into a single output. The idea is that a small convolution filter size identifies local features well but may focus too much on the local regions, whereas large convolution filter has better abstraction but loses local information. The inception module uses various-sized convolution filters and concatenates into a single output to incorporate all these features.
- **Auxilary Classifier**: Another key feature of Inception V1 network is that it has two auxiliary classifiers which contribute to 30% of the final classification loss. The idea is that by adding auxiliary output in the middle of the network, it solves the gradient vanishing issue during backpropagation. The auxilary classifiers are only used in the training phase.
- **Architecture:** Consists of 22 layers deep but with fewer parameters than VGGNet due to the inception modules. There are 1x1 convolutions incorporated in each of the convolutions in the inception module to significantly reduce computational cost and make the model deeper.

**Pros:**

- High performance with fewer parameters and computational cost.
- Innovative inception modules allow for more efficient computations.

**Cons:**

- Complex architecture compared to VGGNet.
- Implementation can be challenging.

## Inception v2 and v3

Original Paper: **Szegedy, Christian, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna.** "Rethinking the Inception Architecture for Computer Vision." *arXiv preprint arXiv:1512.00567* (2015). Available at: https://arxiv.org/abs/1512.00567

- **Improvements:** Incorporate various factorization ideas to reduce the computational cost and improve training.

  ![img](https://blog.kakaocdn.net/dn/bD8NHs/btqZ1JqQ7ZM/zD651R45o7hNN7AgScgpik/img.png)

- **Factorization:** Uses smaller convolutions (e.g., nxn convolution factorized into nx1 and 1xn convolutions) to reduce computational cost. Both 5x5 and 7x7 convolutions are also replaced by two, three 3x3 convolutions.

**Inception v2:**

- Removed the earlier auxiliary classifier as it showed that it is useless.
- Factorized convolutions to further reduce computation.

**Inception v3:**

- Changed optimizer to RMSProp
- No longer use one-hot encoding for the target to prevent overfitting (label smoothing)
- Used batch normalization in the last FC layer