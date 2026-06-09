# Backpropagation and Gradients

## The Core Question

After GPT makes a mistake:

```text
Input:
Every effort moves you

Correct:
forward

Predicted:
backward
```

How does the model know:

```text
Which weight was responsible?
```

and

```text
How should it change?
```

The answer is:

```text
Backpropagation
```

---

# What Is Backpropagation?

Backpropagation is the algorithm used to determine:

```text
How each parameter contributed to the loss.
```

It computes gradients for every trainable parameter in the model.

---

# What Is a Gradient?

A gradient answers:

```text
If I change this parameter slightly,
how will the loss change?
```

Think of a gradient as:

```text
Direction of improvement.
```

---

# Mountain Analogy

Imagine standing on a mountain.

Goal:

```text
Reach the lowest point.
```

Loss Function:

```text
Height of the mountain.
```

Gradient:

```text
Direction of steepest ascent.
```

To minimize loss:

```text
Move opposite the gradient.
```

---

# Example

Suppose:

```text
Weight = 2.0

Gradient = +0.5
```

Positive gradient means:

```text
Increasing weight
increases loss.
```

Therefore:

```text
Decrease weight.
```

---

Suppose:

```text
Gradient = -0.5
```

Negative gradient means:

```text
Increasing weight
decreases loss.
```

Therefore:

```text
Increase weight.
```

---

# Where Do Gradients Come From?

Training pipeline:

```text
Input
 ↓

Forward Pass
 ↓

Prediction
 ↓

Cross Entropy Loss
```

Once loss is computed:

```text
Backpropagation
```

applies the chain rule from calculus to determine:

```text
How much every parameter
affected the loss.
```

---

# What Receives Gradients?

Everything trainable.

Examples:

```text
Token Embeddings

Position Embeddings

Wq

Wk

Wv

FFN Weights

LayerNorm Parameters

Output Layer
```

All receive gradients.

---

# Why Is Backpropagation Necessary?

Without backpropagation:

```text
Model Knows Error
```

but not:

```text
How To Fix Error
```

Backpropagation transforms:

```text
Error
```

into

```text
Weight Updates
```

---

# What Happens After Gradients Are Computed?

The optimizer updates weights.

Example:

```text
new_weight

=

old_weight
-
learning_rate × gradient
```

This moves parameters toward lower loss.

---

# Gradient Flow Through GPT

A simplified view:

```text
Loss
 ↓

Output Layer
 ↓

Transformer Block 12
 ↓

Transformer Block 11
 ↓

...
 ↓

Transformer Block 1
 ↓

Embeddings
```

Gradients flow backward through the entire network.

This is why the algorithm is called:

```text
Backpropagation
```

---

# Why Residual Connections Matter

Deep networks often suffer from:

```text
Vanishing Gradients
```

Residual connections create shortcut paths.

This allows gradients to flow more easily through many layers.

Without residual connections, training modern GPT models would be extremely difficult.

---

# Why LayerNorm Matters

LayerNorm keeps activations stable.

Stable activations produce healthier gradients.

This improves optimization and training stability.

---

# One Sentence Summary

Backpropagation computes gradients that tell every trainable parameter how it should change to reduce the loss, allowing the optimizer to gradually improve the model's predictions.
