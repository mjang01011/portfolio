# Training vs. Validation vs. Test 

In machine learning, evaluating a model's performance accurately is crucial for developing robust and effective models. To achieve this, we typically split our dataset into three distinct subsets: training, validation, and test sets. Each of these subsets serves a specific purpose in the model development process.

![Train, Validation, and Test Set | 포자랩스의 기술 블로그](https://pozalabs.github.io/assets/images/Dataset_Splitting/data_split.jpg)

## 1. **Training Set**

The **training set** is the portion of the dataset used to train the model. This data is used to adjust the model's parameters, learn patterns, and fit the model to the underlying data distribution. 

- **Model Learning:** The model uses the training data to learn and update its parameters. This involves adjusting weights and biases to minimize the error between the predicted and actual outcomes.
- **Optimization:** The optimization algorithm (gradient descent) works to reduce the loss based on the training data.

## 2. **Validation Set**

The **validation set** is a subset of the training data that is set aside to evaluate the model during training. The primary purpose of the validation set is to tune hyperparameters and make decisions about the model's architecture.

- **Hyperparameter Tuning:** Hyperparameters (e.g., learning rate) are adjusted based on the performance of the model on the validation set. This helps in finding the optimal configuration for the model.
- **Model Selection:** The validation set is used to select the best model or checkpoint. We can use the validation results to determine the best model to test as our final model by choosing the model at the epoch where the validation loss is minimum.

## 3. **Test Set**

The **test set** is used to evaluate the final model’s performance after training is complete. This set should be completely separate from both the training and validation sets to ensure that the evaluation is on truly unseen data.

- **Performance Evaluation:** The test set provides an estimate of how the model will perform on new, unseen data in the real world. It is used to gauge the generalization ability of the model.

It is very important to note that no additional learning or tuning is performed based on the test set. The results from the test set give a final assessment of model performance. We can also measure other generalization performance metrics of the model, such as precision, recall, and F1 score.

## Summary

In summary, the different subsets of data serve unique roles:

- **Training Data:** Used to train the model and adjust parameters.
- **Validation Data:** Used to tune hyperparameters and select the best model.
- **Test Data:** Used to evaluate the model’s final performance.

# K-Fold Cross Validation

If we have too few training data to even have a decently sized validation set, we can divide the training data in to 'k' parts and create k different sets of training and validation set pairs. We can calculate the average validation loss across the folds to determine the set of hyperparameters with the lowest loss.