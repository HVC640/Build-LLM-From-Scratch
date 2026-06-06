# Module 7: GELU Activation Function

## What is GELU?

GELU stands for:

```text
Gaussian Error Linear Unit
```

It is the activation function used in GPT models.

---

## Why Do We Need Activation Functions?

Without activations:

```text
Linear
↓
Linear
↓
Linear
```

collapses into a single linear transformation.

The network would be unable to learn complex patterns.

Activation functions introduce non-linearity.

---

## ReLU

ReLU is defined as:

ReLU(x)=max(0,x)

Negative values become:

```text
0
```

Positive values remain unchanged.

---

## Problems with ReLU

### Dead Neurons

If a neuron consistently receives negative values:

```text
Output = 0
```

It may stop learning.

---

### Sharp Cutoff

At:

```text
x = 0
```

the function changes abruptly.

This can make optimization less smooth.

---

## GELU

GELU provides a smooth alternative.

Instead of completely removing negative values:

```text
Negative values are gradually suppressed.
```

Small negative inputs can still contribute information.

---

## Why GPT Uses GELU

Benefits:

* Smooth gradients
* Better optimization
* Better information flow
* Improved training stability
* Better empirical performance

These advantages become more important in very large transformer models.

---

## Intuition

ReLU behaves like:

```text
Allowed
or
Blocked
```

GELU behaves like:

```text
How much should I allow this information through?
```

This softer behavior improves learning.

---

## Key Takeaways

* GELU is GPT's activation function.
* It is smoother than ReLU.
* It avoids many dead-neuron issues.
* It improves optimization in large transformer models.
* It sits inside the Feed Forward Network.
