# Types of Learning in ML

This blog aims to provide a brief overview of the four types of learning in ML: 1) supervised, 2) unsupervised, 3) self-supervised, 4) reinforcement learning.

## Supervised learning

Supervised learning involves training a model on a dataset where each example is paired with an output label, known as ground truth. Types of supervised learning tasks include regression and classification. For instance, if we were to train a model to classify the type of animal in a given image, we would use labeled data where each image is tagged with the corresponding animal type. Cat images might be labeled as class '0' and dog images as class '1'. Supervised learning allows us to perform tasks such as localization (detecting the location of a specific object within an image), segmentation (classifying each pixel in an image), and pose estimation (determining the pose of humans in an image). However, creating these labels can be costly and time-consuming, as it requires manual annotation.

## Unsupervised learning

Unsupervised learning methods, on the other hand, do not rely on ground truth labels. Instead, they aim to find patterns and structures within the data. One of the most well-known unsupervised learning algorithms is k-means clustering, which partitions data into 'k' clusters, each represented by the mean of the data points in the cluster. Another type of unsupervised learning is dimensionality reduction, such as Principal Component Analysis (PCA), which transforms data into a set of linearly uncorrelated variables called principal components, thereby reducing the number of variables while retaining most of the original information. Unsupervised learning is useful for discovering hidden patterns or intrinsic structures in data without the need for labeled training examples.

## Self-Supervised Learning (SSL)

The more data we have, the better our models tend to perform. However, generating all the labels for supervised learning tasks can be very time-consuming. Self-supervised learning (SSL) aims to resolve this issue by defining and solving a different problem apart from the true task we want our model to solve, generating labels from the data itself. There are two main steps to SSL, and we will explain this process using the example of classifying a cat from other animals. Check the paper in the references for the original paper for this example.

1. **Pre-train the model with a pretext task**

   ![image-20240520152141355](https://github.com/mjang01011/portfolio/blob/main/public/blogs/markdowns/images/image-20240520152141355.png?raw=true)

   In this step, a random patch (blue rectangle) is chosen, and eight other patch locations around the blue patch are defined. We train our model so that, given a pair of the blue and one of the red patches, it predicts the correct location of the red patch. This task does not directly teach the model to classify a cat from other animals. However, it trains the model to understand the relative locations of a cat's features. The model trained on such a pretext task is called a pre-trained model. This pre-training helps in the next step of correctly classifying cat images.

2. **Perform transfer learning to solve the downstream task**

   Now, we train the model with labeled data so that it correctly classifies the animal type. The intuition is that the pretext task allows our pre-trained model to learn meaningful representations of a cat, reducing the amount of data needed to differentiate cats from other animals. We can adjust the final parts of our model, such as appending a fully connected classification layer, to output one of the N animal classes and use standard supervised learning methods to learn the classification layer weights.

In 2020, contrastive learning has been proposed as a useful pretext task. I have experience examining two SSL methods: SimCLR (Simple Contrastive Learning) and RotNet (Rotation Net). The paper, poster, and code repository can be found [here](https://github.com/mjang01011/Duke_ECE661_Final-Project-An_Evaluation_of_Self-Supervised_Learning_Method-SimCLR_and_RotNet).

## Reinforcement Learning

Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative rewards. This concept can be illustrated with the example of training a dog.

### Key Components of Reinforcement Learning

1. **Agent**: The entity that is learning and making decisions. In our example, the agent is the dog.
2. **Environment**: The world with which the agent interacts. For the dog, this is its surroundings, including the house, yard, and any other place where training occurs.
3. **Actions**: The set of all possible moves the agent can make. For the dog, actions include sitting, standing, barking, fetching, etc.
4. **States**: The current situation or context of the agent in the environment. A state could be the dog being in the kitchen, seeing a toy, or hearing a command.
5. **Rewards**: Feedback from the environment based on the action taken. Positive rewards (treats, praise) encourage the dog to repeat an action, while negative rewards (withholding treats, a stern "no") discourage it.

### The Process of Reinforcement Learning

1. **Initialization**: The dog starts with no knowledge of what actions lead to rewards. It explores its environment, trying different actions.
2. **Action and Feedback**: When the dog performs an action (e.g., sitting when told), it receives feedback. If the dog sits and receives a treat (positive reward), it learns that sitting when commanded is a good action.
3. **Learning from Feedback**: The dog uses the feedback to update its behavior. The goal is to maximize the cumulative rewards. Over time, the dog associates specific actions with positive rewards and is more likely to repeat those actions.
4. **Exploration vs. Exploitation**: The dog has to balance exploring new behaviors (exploration) and using known rewarding behaviors (exploitation). For example, if the dog learns that sitting earns treats, it may initially try other behaviors (like lying down or barking) to see if they also yield rewards.
5. **Policy Development**: The dog develops a policy, which is a strategy or set of rules dictating the best action to take in each state to maximize rewards. For instance, the dog learns to sit when it hears the command "sit" because it knows it will get a treat.

Over time, the dog will sit reliably when it hears the command because it has learned that this action maximizes its rewards. This process demonstrates how reinforcement learning can be used to train an agent—in this case, a dog—by providing feedback that helps it learn the best actions to achieve desired outcomes.

## References

1. Doersch, Carl, Abhinav Gupta, and Alexei A. Efros. "Unsupervised Visual Representation Learning by Context Prediction." *CoRR* abs/1505.05192 (2015). [http://arxiv.org/abs/1505.05192](http://arxiv.org/abs/1505.05192).
2. Sutton, Richard S., and Andrew G. Barto. *Reinforcement Learning: An Introduction*. MIT Press, 2018.