# Training vs Test vs Validation

We want our model to do well on an unseen dataset. Therefore, it is very important that we do not use our test data to train our model. The test set should only be used to determine how well our model performs when inputted with unseen dataset. What if we want to test on unseen dataset while training? This is the purpose of the validation set. We take a part of the training data as the validation set. Overall, we can say that training data is for learning the parameters. Test data is for testing the final version of our model. Validation data is for choosing the right hyperparameters for our model. We can also use the validation results to determine the best model to use as our final model by choosing the model at the epoch where the validation loss is minimum.

# K-Fold Cross Validation

If we have too few training data to even have a decently sized validation set, we can divide the training data in to 'k' parts and create k different sets of training and validation set pairs. We can calculate the average validation loss across the folds to determine the set of hyperparameters with the lowest loss.