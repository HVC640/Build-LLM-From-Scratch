# Module 13: GPT Training Loop

## The Big Picture

Training is simply:

```text id="ig6ynq"
Predict
↓
Measure Error
↓
Update Weights
↓
Repeat
```

The entire GPT training process follows this cycle.

---

## Initial State

At the beginning:

```text id="jjlwmq"
All Weights Are Random
```

Examples:

```text id="93gfjx"
Wq
Wk
Wv

Embeddings

FFN Weights

Output Layer Weights
```

The model produces essentially random predictions.

---

## Step 1: Sample a Batch

Training data:

```text id="9hjs5j"
Every effort moves you forward
```

Example batch:

Input:

```text id="qwyzhx"
[1, 2, 3, 4]
```

Target:

```text id="e9o0bz"
[2, 3, 4, 5]
```

In practice:

```text id="4nngj7"
Many Sequences
```

are processed simultaneously.

---

## Step 2: Forward Pass

Input tokens travel through:

```text id="p9j6xu"
Embeddings
↓
Transformer Blocks
↓
Output Layer
```

The model produces:

```text id="ynkzlw"
Logits
```

for every token position.

---

## Step 3: Convert Logits to Probabilities

SoftMax converts:

```text id="80wjlwm"
Logits
```

into:

```text id="5a1xlu"
Probability Distribution
```

across the vocabulary.

Example:

```text id="7gcj2y"
forward → 0.70

backward → 0.10

quickly → 0.05
```

---

## Step 4: Compute Loss

Cross Entropy compares:

```text id="vuktxu"
Predicted Probability
```

against

```text id="jlwm0x"
Correct Target Token
```

Result:

```text id="gvt78g"
Loss
```

A single number representing prediction quality.

---

## Step 5: Backpropagation

Now the model asks:

```text id="4azp7q"
Which parameters caused this error?
```

Backpropagation computes gradients for:

```text id="8i8xyu"
Embeddings

Attention Weights

FFN Weights

Output Layer
```

Every trainable parameter receives a gradient.

---

## What Is a Gradient?

A gradient tells us:

```text id="uq5e0e"
How should this parameter change
to reduce loss?
```

Think of it as:

```text id="dt4jwu"
Direction of Improvement
```

for each weight.

---

## Step 6: Optimizer Update

GPT commonly uses:

```text id="jlwm41"
AdamW
```

The optimizer updates:

```text id="jlwm42"
weight
=
weight - learning_rate × gradient
```

for millions or billions of parameters.

---

## Step 7: Repeat

After one update:

```text id="7q6n0h"
Training Is Not Finished
```

The process repeats:

```text id="jlwm43"
Next Batch
↓
Forward Pass
↓
Loss
↓
Backpropagation
↓
Optimizer Update
```

thousands or millions of times.

---

## Epoch

An epoch means:

```text id="jlwm44"
One Complete Pass
Through The Entire Dataset
```

Example:

```text id="jlwm45"
Dataset = 1000 batches

1 Epoch = 1000 updates
```

---

## What Happens During Training?

Initially:

```text id="jlwm46"
Loss = High
Perplexity = High
```

Predictions are poor.

---

After many updates:

```text id="jlwm47"
Loss ↓

Perplexity ↓
```

Predictions improve.

---

## Why Does Training Work?

The model repeatedly adjusts weights to increase:

```text id="jlwm48"
P(correct token)
```

and decrease:

```text id="jlwm49"
P(incorrect tokens)
```

Eventually the parameters encode useful language patterns.

---

## Complete Training Pipeline

```text id="jlwm50"
Dataset
   ↓

Batch Sampling
   ↓

Forward Pass
   ↓

Logits
   ↓

SoftMax
   ↓

Cross Entropy Loss
   ↓

Backpropagation
   ↓

Gradients
   ↓

AdamW Update
   ↓

New Weights
   ↓

Repeat
```

---

## Mental Model

Imagine teaching a student.

```text id="jlwm51"
Ask Question
↓

Student Answers
↓

Check Answer
↓

Give Feedback
↓

Student Adjusts
↓

Ask Next Question
```

Training GPT follows exactly the same idea.

---

## Key Takeaways

* Training consists of repeated prediction and correction.
* Forward pass generates predictions.
* Cross Entropy measures error.
* Backpropagation computes gradients.
* AdamW updates weights.
* Loss and perplexity should decrease during training.
* Repeating this loop millions of times creates a capable language model.
