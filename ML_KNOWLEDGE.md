# Machine Learning and Neural Networks

## What is Machine Learning?

Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data, identify patterns, and make decisions with minimal human intervention.

**Core purpose of machine learning:**

- Automatically discover patterns in data
- Use these patterns to predict future data or make decisions
- Improve performance with experience/more data
- Handle tasks that are too complex for traditional programming

### Types of Machine Learning

#### 1. Supervised Learning

- Learns from labeled examples (input → correct output)
- Goal: Predict outputs for new, unseen inputs
- **Common applications**: Classification, regression, forecasting
- **Examples in finance**: Credit scoring, stock price prediction, fraud detection

#### 2. Unsupervised Learning

- Works with unlabeled data
- Goal: Find structure, patterns, or relationships in data
- **Common applications**: Clustering, dimensionality reduction, anomaly detection
- **Examples in finance**: Customer segmentation, portfolio diversification, market structure analysis

#### 3. Reinforcement Learning

- Agent learns by interacting with an environment
- Goal: Learn a policy that maximizes cumulative rewards
- **Common applications**: Game playing, robotics, resource management
- **Examples in finance**: Algorithmic trading, portfolio optimization

### Common Machine Learning Algorithms

#### Classification (Supervised Learning)

- **Definition:** Classification is a supervised learning technique where the goal is to assign input data to one of several predefined categories or classes.
- **Example:** Email spam detection (spam vs. not spam), image recognition (cat, dog, car, etc.), or, in your project, classifying investment text into risk levels, investment horizons, and actions.
- **Common algorithms:** Logistic regression, decision trees, random forests, support vector machines (SVM), neural networks.
- **Supervised/Unsupervised:** Supervised (requires labeled data with known class outputs).

#### Classification Algorithms

- **Logistic Regression**: Simple linear model for binary/multi-class classification
- **Decision Trees**: Flowchart-like structure for decisions based on feature values
- **Random Forests**: Ensemble of decision trees for improved accuracy and robustness
- **Support Vector Machines (SVM)**: Finds optimal boundary between classes
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **K-Nearest Neighbors (KNN)**: Classifies based on majority vote of nearest neighbors

#### Regression (Supervised Learning)

- **Definition:** Regression is a supervised learning technique where the goal is to predict a continuous numeric value (not a category) based on input features.
- **Example:** Predicting house prices based on size, location, and number of bedrooms.
- **Common algorithms:** Linear regression, polynomial regression, ridge/lasso regression.
- **Supervised/Unsupervised:** Supervised (requires labeled data with known outputs).

#### Regression Algorithms

- **Linear Regression**: Models linear relationship between dependent and independent variables
- **Polynomial Regression**: Extension of linear regression for non-linear relationships
- **Ridge/Lasso Regression**: Linear regression with regularization to prevent overfitting
- **Decision Tree Regression**: Uses decision trees for predicting continuous values
- **Gradient Boosting**: Ensemble technique building trees sequentially to correct errors

#### Clustering (Unsupervised Learning)

- **Definition:** Clustering is an unsupervised learning technique where the goal is to group similar data points together based on their features, without using labels.
- **Example:** Grouping customers into market segments based on their purchasing behavior.
- **Common algorithms:** K-Means, hierarchical clustering, DBSCAN.
- **Supervised/Unsupervised:** Unsupervised (does not require labeled data).

#### Clustering Algorithms

- **K-Means**: Partitions data into K clusters based on feature similarity
- **Hierarchical Clustering**: Builds nested clusters by merging or splitting
- **DBSCAN**: Density-based clustering for finding arbitrary-shaped clusters

#### Dimensionality Reduction (Unsupervised Learning)

- **Definition:** Dimensionality reduction is the process of reducing the number of features (variables) in your data while preserving as much information as possible.
- **Purpose:** Makes data easier to visualize, speeds up training, and can help remove noise.
- **Example:** Reducing a dataset with 100 features down to 2 or 3 for visualization.
- **Common algorithms:** Principal Component Analysis (PCA), t-SNE, autoencoders.
- **Supervised/Unsupervised:** Unsupervised (typically does not require labeled data).

#### Dimensionality Reduction

- **Principal Component Analysis (PCA)**: Reduces dimensions while preserving variance
- **t-SNE**: Visualizes high-dimensional data in 2D or 3D space
- **Autoencoders**: Neural networks that compress then reconstruct data

#### Sequence Models

- **Recurrent Neural Networks (RNN)**: Process sequential data with memory
- **Long Short-Term Memory (LSTM)**: Special RNN that addresses vanishing gradient problem
- **Gated Recurrent Units (GRU)**: Simplified version of LSTM with fewer parameters

### Machine Learning Workflow

1. **Problem Definition**: Define the task and success metrics
2. **Data Collection**: Gather relevant data for the task
3. **Data Preprocessing**: Clean, normalize, and transform data
4. **Feature Engineering**: Create meaningful features from raw data
5. **Model Selection**: Choose appropriate algorithm(s) for the task
6. **Model Training**: Fit the model to the training data
7. **Model Evaluation**: Assess performance on validation data
8. **Hyperparameter Tuning**: Optimize model parameters
9. **Model Deployment**: Implement the model in production
10. **Monitoring and Maintenance**: Track performance and retrain as needed

### Features in Machine Learning

- **Definition:** A feature is an individual measurable property or characteristic of the data that is used as input to a machine learning model.
- **Purpose:** Features are the pieces of information the model uses to make predictions or classifications.
- **Examples:**
  - In a house price prediction dataset: features could be square footage, number of bedrooms, location, and year built.
  - In an email spam classifier: features could be the presence of certain keywords, the length of the email, or the sender's address.
  - In your investment text classifier: features are the tokenized words or subwords extracted from the investment text.
- **Feature Engineering:** The process of selecting, modifying, or creating new features to improve model performance.

### Training, Validation, and Test Sets in Machine Learning

- **Training set:** Used to fit the model's parameters. The model learns patterns from this data.
- **Validation set:** Used to tune hyperparameters and monitor performance during training. Helps detect overfitting and guides model selection.
- **Test set:** Used only after training and tuning are complete, to evaluate the model's final performance on new, unseen data. Measures how well the model generalizes.

**Why split data this way?**

- Ensures the model is evaluated fairly on data it hasn't seen before
- Helps prevent overfitting and underfitting
- Provides a realistic estimate of real-world performance

### Ensemble Learning, Bagging, and Boosting

#### Ensemble Learning

- **Definition:** Ensemble learning combines multiple models to produce a better overall result than any single model could achieve alone. The idea is that a group of "weak" models can come together to form a "strong" model.

#### Bagging (Bootstrap Aggregating)

- **How it works:** Multiple models (often the same type, like decision trees) are trained independently on different random subsets of the training data (with replacement).
- **Prediction:** For classification, the final prediction is usually the majority vote; for regression, it's the average.
- **Goal:** Reduce variance and help prevent overfitting.
- **Example:** Random Forest is a classic bagging algorithm (it builds many decision trees on random data samples and averages their results).

#### Boosting

- **How it works:** Models are trained sequentially. Each new model tries to correct the mistakes of the previous one by focusing more on the data points that were misclassified.
- **Prediction:** The final prediction is a weighted combination of all models' outputs.
- **Goal:** Reduce both bias and variance, and build a strong model from many weak ones.
- **Example:** AdaBoost and Gradient Boosting Machines (GBM, XGBoost, LightGBM) are popular boosting algorithms.

#### Key Differences

- **Bagging:** Models are trained independently and in parallel; reduces variance.
- **Boosting:** Models are trained sequentially, each learning from the previous; reduces bias and variance.

| Technique | Training Style | Combines | Example       | Main Benefit            |
| --------- | -------------- | -------- | ------------- | ----------------------- |
| Bagging   | Parallel       | Votes    | Random Forest | Reduces variance        |
| Boosting  | Sequential     | Weighted | AdaBoost, XGB | Reduces bias & variance |

### Overfitting, Underfitting, and Generalization

#### Overfitting

- **Definition:** Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor performance on new, unseen data.
- **Symptoms:** High accuracy on training data but low accuracy on validation/test data.
- **Causes:** Model is too complex, too many parameters, not enough training data, or training for too many epochs.
- **Prevention Techniques:**
  - Use more training data
  - Apply regularization (L1/L2, dropout)
  - Use simpler models
  - Cross-validation
  - Early stopping
  - Data augmentation (for images/text)

#### Underfitting

- **Definition:** Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and validation data.
- **Symptoms:** Low accuracy on both training and validation/test data.
- **Causes:** Model is too simple, not enough features, insufficient training, or too much regularization.
- **Prevention Techniques:**
  - Use a more complex model
  - Add more relevant features
  - Train for more epochs
  - Reduce regularization

#### Generalization

- **Definition:** Generalization is the model's ability to perform well on new, unseen data (not just the data it was trained on).
- **Goal:** Achieve a balance between underfitting and overfitting so the model captures the true patterns in the data and can make accurate predictions on new data.

### Machine Learning vs. Deep Learning

- **Machine Learning**: Broader field encompassing many algorithms and approaches
- **Deep Learning**: Subset of ML focusing on neural networks with multiple layers
  - Excels at learning hierarchical features from data
  - Often outperforms traditional ML for unstructured data (images, text, audio)
  - Requires more data and computational resources
  - Less interpretable than some traditional ML models

Our investment text classifier uses deep learning (specifically transformers) because of its superior ability to understand the nuances and contextual relationships in text data.

# Understanding Neural Networks

This document provides an overview of neural networks, how they work, and their application in our investment text classifier.

## What is a Neural Network?

A neural network is a computational model inspired by the human brain's structure and function. These networks are designed to recognize patterns in data through a process resembling how human neurons process information.

**Key characteristics:**

- Composed of interconnected processing nodes (neurons)
- Learn from examples rather than being explicitly programmed
- Improve performance over time through training
- Excel at finding patterns in complex, high-dimensional data

## Neural Network Layers

Neural networks are organized in layers of neurons. Each layer performs specific transformations on the data:

### 1. Input Layer

- Receives the raw data (in our model, tokenized investment text)
- Doesn't perform computations, just passes data to the next layer
- For text data, this often represents numerical token IDs

### 2. Hidden Layers

Our investment text classifier uses DistilBERT, which contains multiple sophisticated hidden layers:

#### Embedding Layers

- Convert token IDs into dense vectors (typically 768 dimensions for DistilBERT)
- Capture semantic meaning of words
- Similar words have similar embeddings in the vector space

#### Transformer Layers

- **Self-Attention Mechanism**: Helps the model understand relationships between words in context
  - Example: In "Buy tech stocks with strong growth," attention helps link "strong" with "growth"
  - Each word attends to all other words with different weights
- **Feed-Forward Networks**: Process the outputs of attention
  - Apply non-linear transformations
  - Learn complex patterns in the data
- **Layer Normalization**: Stabilizes the learning process
  - Normalizes the activations to prevent extreme values
  - Helps the model train faster and more reliably

### What is a Transformer in Machine Learning?

A **transformer** is a type of deep learning model architecture that has revolutionized natural language processing (NLP) and many other fields. It was introduced in the 2017 paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

**Key Concepts:**

- **Self-Attention Mechanism:**
  - The core innovation of transformers. It allows the model to weigh the importance of different words in a sentence, regardless of their position. For example, in the sentence "The cat, which was black, sat on the mat," the model can relate "cat" and "sat" even though they are far apart.
- **Parallelization:**
  - Unlike older models (like RNNs), transformers process all words in a sequence at once, making them much faster to train.
- **Stacked Layers:**
  - Transformers are built from multiple identical layers, each containing self-attention and feed-forward sub-layers.

**Why Are Transformers Important?**

- **State-of-the-art Results:**
  - Transformers power models like BERT, GPT, and DistilBERT, which achieve top results in tasks like text classification, translation, and question answering.
- **Versatility:**
  - They are used not just in NLP, but also in computer vision, audio, and even reinforcement learning.

**How Do Transformers Work?**

1. **Input Embedding:** Each word is converted to a vector (embedding).
2. **Self-Attention:** The model computes attention scores for every word pair, learning which words are most relevant to each other.
3. **Feed-Forward Layers:** The attended information is processed through neural network layers.
4. **Output:** The final layer produces predictions (e.g., classifying text, generating text, etc.).

**In Your Project**

Your investment text classifier uses a transformer (DistilBERT) to understand the context and meaning of financial texts, enabling it to make accurate multi-label predictions.

### 3. Output Layers

Our model has three specialized output layers (called "heads"):

- **Risk Level Classifier**: 3 neurons (high/medium/low)
- **Investment Horizon Classifier**: 3 neurons (short/medium/long-term)
- **Action Classifier**: 4 neurons (buy/sell/hold/ignore)

Each output neuron produces a score, and the highest score in each category represents the model's prediction.

## How Information Flows Through a Neural Network

1. **Input Processing**: Text tokens enter the network
2. **Forward Propagation**: Each layer transforms the data by:
   - Multiplying inputs by weights
   - Adding biases
   - Applying activation functions (e.g., ReLU, GELU, or sigmoid)
3. **Feature Extraction**: Hidden layers extract increasingly abstract features
   - Early layers: basic patterns
   - Deeper layers: complex concepts
4. **Output Generation**: Final layers produce probability scores for each class

## The Learning Process

Neural networks learn through a process called training:

1. **Initialization**: The model starts with random weights
2. **Forward Pass**: The model makes predictions on training data
3. **Loss Calculation**: Predictions are compared to correct answers using a loss function
   - In our model, we use Cross-Entropy Loss for each classification task
4. **Backpropagation**: The error is propagated backward through the network
5. **Weight Updates**: The model's parameters are adjusted to reduce errors
   - Our model uses the AdamW optimizer, which adapts learning rates for different parameters
6. **Iteration**: Steps 2-5 repeat for multiple epochs until performance stops improving

## Multi-Task Learning

Our investment text classifier employs multi-task learning, where a single model learns to perform multiple related tasks simultaneously:

- Shares a common representation (the transformer layers)
- Has task-specific output layers
- Benefits from knowledge transfer between tasks
- Typically results in better generalization than separate models

## Transformer Architecture

The transformer architecture used in our model has several advantages over traditional RNNs (Recurrent Neural Networks):

- **Parallelization**: Can process all tokens simultaneously
- **Long-range Dependencies**: Better captures relationships between distant words
- **Pre-training**: Leverages knowledge from massive text corpora
- **Attention Mechanisms**: Focuses on relevant parts of the input

## Practical Considerations

When working with neural networks like the one in our investment text classifier:

- **Training Data**: More diverse and representative data generally leads to better performance
- **Hyperparameters**: Learning rate, batch size, and model size affect training dynamics
- **Overfitting**: Too much specialization on training data can hurt generalization
- **Evaluation**: Always validate on separate data to ensure real-world performance

## Further Learning Resources

To deepen your understanding of neural networks:

- [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Documentation for the library we use
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Free online book by Michael Nielsen
