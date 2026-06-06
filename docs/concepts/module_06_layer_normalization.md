# Module 6: Layer Normalization

## Why Do We Need Layer Normalization?

GPT models are extremely deep.

Examples:

```text
GPT-2 Small   -> 12 layers
GPT-2 Large   -> 36 layers
GPT-3         -> 96+ layers
```

As information flows through many layers, activations and gradients can become unstable.

This creates two major problems:

1. Vanishing Gradients
2. Exploding Gradients

Layer Normalization helps stabilize training by keeping activations within a predictable range.

---

## Vanishing and Exploding Gradients

During backpropagation:

```text
Loss
  ↓
Layer N
  ↓
Layer N-1
  ↓
...
```

Gradients are repeatedly multiplied.

If values are:

```text
Very Small
```

gradients shrink toward zero.

If values are:

```text
Very Large
```

gradients explode.

Both situations make learning difficult.

Layer Normalization reduces this instability.

---

## Internal Covariate Shift

As training progresses:

```text
Weights Change
      ↓
Activation Distribution Changes
      ↓
Later Layers Must Constantly Adapt
```

This phenomenon is called Internal Covariate Shift.

Layer Normalization stabilizes activation distributions and makes optimization easier.

---

## How LayerNorm Works

For each token independently:

### Step 1

Compute Mean

```text
mean(x)
```

---

### Step 2

Compute Variance

```text
var(x)
```

---

### Step 3

Normalize

```text
(x - mean)
------------------
sqrt(var + eps)
```

The result has:

```text
Mean ≈ 0
Variance ≈ 1
```

---

## Why Add Epsilon?

```text
eps = 1e-5
```

or similar.

Purpose:

```text
Prevent Division by Zero
```

when variance becomes extremely small.

---

## Learnable Parameters

LayerNorm contains:

```text
Scale (γ)
Shift (β)
```

These parameters are trainable.

Final output:

```text
γ * normalized_x + β
```

This allows the model to learn the most useful scaling and shifting for the task.

---

## LayerNorm in GPT

LayerNorm operates across:

```text
Embedding Dimension
```

For GPT-2 Small:

```text
768 dimensions
```

Each token is normalized independently.

---

## Why LayerNorm is Important

Benefits:

* Stable training
* Faster convergence
* Better gradient flow
* Enables deeper networks
* Reduces training instability

Without LayerNorm, modern GPT models would be significantly harder to train.

---

## Key Takeaways

* LayerNorm normalizes activations.
* Prevents exploding and vanishing gradients.
* Stabilizes training in deep transformers.
* Uses learnable scale and shift parameters.
* Operates across the embedding dimension.
