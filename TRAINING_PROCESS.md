# Training Process for the Investment Text Classifier

This document explains how the investment text classifier model is trained, from data preparation to model evaluation.

## 1. Data Preparation

The model learns from labeled examples stored in CSV files:

- **Training data**: The main dataset the model learns from (`data/training/training_data.csv`)
- **Validation data**: Used to monitor performance during training (`data/validation/validation_data.csv`)
- **Test data**: Used for final evaluation after training (`data/test/test_data.csv`)

Each example contains:

- An investment-related text (advice, analysis, recommendation)
- Three labels:
  - Risk level (high/medium/low)
  - Investment horizon (short-term/medium-term/long-term)
  - Action recommendation (buy/sell/hold/ignore)

## 2. Model Architecture

The model is not a simple linear algorithm, but rather a neural network with multiple components:

- **Base Model**: DistilBERT, a pre-trained transformer model that understands language context
- **Task-specific Heads**: Three separate neural network layers that make predictions for:
  - Risk level classification
  - Investment horizon classification
  - Action recommendation classification

This is a multi-task learning approach where the model learns to solve multiple classification problems simultaneously.

## 3. Training Algorithm

While the training process runs sequentially (one batch after another), the underlying algorithm is a complex neural network:

1. **Not Linear**: Unlike linear algorithms (like linear regression), this model uses non-linear activation functions and complex attention mechanisms from the transformer architecture.

2. **Parallel Processing**: The actual computations leverage parallel processing on the CPU/GPU, even though the training loop executes sequentially.

3. **Gradient Descent**: The model learns through backpropagation and gradient descent, iteratively adjusting weights to minimize prediction errors.

4. **Optimization**: The training uses AdamW optimizer, which adapts learning rates for different parameters based on historical gradient information.

## 4. Training Process Steps

The training process follows these steps:

1. **Data Loading**:

   - Load and preprocess the training and validation data
   - Convert texts to numerical token sequences
   - Prepare data loaders for batched processing

2. **Model Initialization**:

   - Initialize the DistilBERT model with pre-trained weights
   - Add classification heads for all three tasks
   - Move the model to available hardware (CPU/GPU)

3. **Training Loop** (for each epoch):

   - **Forward Pass**: Process batches of text through the model to get predictions
   - **Loss Calculation**: Compute how wrong the predictions are compared to true labels
   - **Backward Pass**: Calculate gradients to determine how to adjust model weights
   - **Optimization Step**: Update model weights to improve predictions
   - **Learning Rate Scheduling**: Adjust learning rates based on progress

4. **Validation**:

   - After each epoch, evaluate model on validation data
   - Track metrics like accuracy and F1 score for each task
   - Save the best-performing model based on validation loss

5. **Model Saving**:
   - Save the best model weights to disk
   - Save the tokenizer for later use in inference

## 5. Running the Training

To start the training process:

```bash
python -m app.models.train --data-path data/training/training_data.csv --validation-path data/validation/validation_data.csv
```

You can customize the training with additional parameters:

- `--epochs`: Number of training iterations (default: 5)
- `--batch-size`: Number of examples processed together (default: 16)
- `--learning-rate`: Controls step size in optimization (default: 2e-5)

## 6. After Training

Once training is complete:

- The model is saved to `models/classifier.pt`
- The tokenizer is saved to `models/tokenizer`
- You can evaluate the model on test data to measure its generalization capability
- The model can be used to classify new investment texts

## Summary

The training process for this model is significantly more complex than linear algorithms. It uses deep learning with transformer neural networks to understand the nuances of investment text and provide multi-faceted classification. While the code executes steps sequentially, the underlying computations involve parallel processing, non-linear functions, and sophisticated optimization techniques.
