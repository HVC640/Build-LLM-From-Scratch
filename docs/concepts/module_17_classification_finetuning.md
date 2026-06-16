# Module 17: Classification Fine-Tuning

## What is Classification Fine-Tuning?

Classification fine-tuning adapts a pretrained GPT model to predict a fixed set of labels.

Examples:

```text
Spam Detection

Sentiment Analysis

Intent Classification

Topic Classification

Document Categorization
```

Instead of predicting:

```text
Next Token
```

the model predicts:

```text
Class Label
```

---

# Why Fine-Tune Instead of Training From Scratch?

Pretrained GPT already understands:

* Language
* Grammar
* Semantics
* Context

We only need to teach it:

```text
How to map text → label
```

for a specific task.

---

# Classification Pipeline

## Stage 1: Dataset Preparation

### Download Dataset

Example:

```text
Spam
Not Spam
```

---

### Preprocess Dataset

Convert:

```text
Text
```

into:

```text
Token IDs
```

---

### Create DataLoaders

Batch the data for efficient training.

---

# Stage 2: Model Setup

## Initialize GPT

Load the pretrained GPT architecture.

---

## Load Pretrained Weights

Import the knowledge learned during pretraining.

---

## Replace Output Layer

This is the most important modification.

Pretraining output:

```text
Vocabulary Size
```

Example:

```text
768 → 50,257
```

---

Classification output:

```text
768 → Number of Classes
```

Example:

```text
768 → 2
```

for:

```text
Spam
Not Spam
```

---

# Why Replace The Output Layer?

During pretraining:

```text
Predict Next Token
```

During classification:

```text
Predict Class
```

The objectives are different.

Therefore the final layer must change.

---

# Training Objective

Instead of:

```text
Next Token Prediction
```

the model learns:

```text
Text → Label
```

mapping.

Cross Entropy Loss is still commonly used.

---

# Inference

Input:

```text
This offer is too good to be true!
```

Output:

```text
Spam
```

---

# Key Takeaways

* Classification fine-tuning changes GPT into a classifier.
* The output layer is replaced with a class-specific layer.
* The pretrained language understanding remains useful.
* Only the task objective changes.
