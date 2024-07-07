# Language Detection Project Summary

## Models and Results

### Logistic Regression
Logistic Regression is a fundamental linear model used for classification tasks. It models the probability of a certain class using the logistic function.

- **Description:**
  Logistic Regression is particularly suitable for tasks where the relationship between features and the target variable is linear. It's efficient to train and interpret, making it a good baseline model for text classification.

- **Accuracy:** 92.89%

### Feed Forward Neural Network
A Feed Forward Neural Network (FFNN) is a versatile neural network architecture that consists of multiple layers of neurons, including input, hidden, and output layers. It processes data in a forward direction from input to output.

- **Description:**
  FFNNs are capable of learning complex patterns and relationships in data, making them suitable for tasks where features are non-linearly related to the target variable. They require more computational resources for training compared to linear models like Logistic Regression.

- **Accuracy:** 93.09%

### Naive Bayes Classifier
The Naive Bayes Classifier is a probabilistic model based on Bayes' theorem with strong independence assumptions between features.

- **Description:**
  Naive Bayes classifiers are simple yet effective for text classification tasks, especially when working with large feature spaces. Despite their naive assumption of feature independence, they often perform well in practice.

- **Classification Report on Test Data:**

| Language   | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Arabic     | 1.00      | 0.98   | 0.99     | 106     |
| Danish     | 0.97      | 0.96   | 0.97     | 73      |
| Dutch      | 0.99      | 0.97   | 0.98     | 111     |
| English    | 0.92      | 1.00   | 0.96     | 291     |
| French     | 0.99      | 0.99   | 0.99     | 219     |
| German     | 1.00      | 0.97   | 0.98     | 93      |
| Greek      | 1.00      | 0.97   | 0.99     | 68      |
| Hindi      | 1.00      | 1.00   | 1.00     | 10      |
| Italian    | 1.00      | 0.99   | 1.00     | 145     |
| Kannada    | 1.00      | 1.00   | 1.00     | 66      |
| Malayalam  | 1.00      | 0.98   | 0.99     | 121     |
| Portuguese | 0.99      | 0.98   | 0.99     | 144     |
| Russian    | 1.00      | 0.99   | 0.99     | 136     |
| Spanish    | 0.99      | 0.97   | 0.98     | 160     |
| Swedish    | 1.00      | 0.98   | 0.99     | 133     |
| Tamil      | 1.00      | 0.99   | 0.99     | 87      |
| Turkish    | 1.00      | 0.94   | 0.97     | 105     |

- **Overall Accuracy:** 98%

### Conclusion
In this language detection project, we employed three different algorithms: Logistic Regression, Feed Forward Neural Network, and Naive Bayes Classifier. Each algorithm was evaluated based on its accuracy and performance across various languages.

- **Logistic Regression** achieved an accuracy of 92.89%, showcasing its effectiveness as a straightforward linear model for text classification.

- **Feed Forward Neural Network** performed slightly better with an accuracy of 93.09%, leveraging its capability to learn intricate patterns in data.

- **Naive Bayes Classifier** outperformed both with an overall accuracy of 98%. Its detailed classification report demonstrates high precision, recall, and F1-scores across multiple languages, highlighting its robustness in multilingual text classification tasks.

These results provide valuable insights into the strengths and limitations of each algorithm in the context of language detection, offering a foundation for further research and application in natural language processing.
