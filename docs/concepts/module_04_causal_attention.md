# Module 4: Causal Attention (Masked Attention)

## The Problem

Standard self-attention allows every token to see every other token.

Example:

```text
I love machine learning
```

When processing:

```text
love
```

the model can already see:

```text
machine
learning
```

This creates a problem during language model training.

The model can "cheat" by looking at future tokens.

This phenomenon is called information leakage.

---

## What is Causal Attention?

Causal Attention restricts each token to attend only to:

* Previous tokens
* Current token

and prevents access to:

* Future tokens

This ensures autoregressive generation.

---

## Why GPT Uses Causal Attention

GPT generates text one token at a time.

Example:

```text
The sky is
```

The model predicts:

```text
blue
```

The model must not see future words while making the prediction.

Causal attention enforces this behavior during training.

---

## Causal Mask

A mask is applied to the attention score matrix.

Example:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Tokens can only attend to positions on or below the diagonal.

---

## Preferred Implementation

Instead of:

```text
SoftMax
↓
Mask
↓
Normalize Again
```

Modern implementations use:

```text
Attention Scores
↓
Add -∞ Mask
↓
SoftMax
```

---

## Why Use -∞ ?

Example:

```text
[2.1, 1.8, -∞]
```

After SoftMax:

```text
[0.57, 0.43, 0]
```

Future positions receive exactly zero attention.

This completely prevents information leakage.

---

## Dropout in Attention

Attention layers can overfit.

Some attention connections may dominate training.

To improve generalization, dropout is applied.

---

## What Does Dropout Do?

Randomly removes some connections during training.

Example:

```text
0.30
0.25
0.20
0.15
0.10
```

might become:

```text
0.30
0
0.20
0
0.10
```

---

## Why Rescale?

When dropout is applied:

```text
keep probability = 1 - p
```

Remaining values are scaled by:

```text
1 / (1 - p)
```

This preserves the expected activation magnitude.

---

## Key Takeaways

* Causal attention prevents future-token access.
* It enables autoregressive generation.
* GPT relies on causal masking.
* Modern implementations apply masking before SoftMax.
* Using -∞ ensures masked positions receive zero probability.
* Dropout improves generalization and reduces overfitting.
