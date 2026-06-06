# Module 8: Residual (Shortcut) Connections

## The Problem

Deep neural networks are difficult to train.

As depth increases:

* Gradients become weaker
* Learning slows down
* Earlier layers receive less useful updates

This is known as the vanishing gradient problem.

---

## What is a Residual Connection?

A residual connection allows information to bypass a layer.

Instead of:

```text
Output = F(x)
```

we compute:

```text
Output = x + F(x)
```

The original input is directly added back.

---

## Why Does This Help?

The model no longer needs to learn:

```text
Entire Transformation
```

Instead it learns:

```text
Difference from Input
```

This is usually easier.

---

## Gradient Flow

Residual connections create a direct path:

```text
Loss
 ↓
Layer
 ↓
Input
```

Gradients can flow through this shortcut without being repeatedly transformed.

This significantly improves training stability.

---

## Residual Connections in GPT

Every transformer block contains:

```text
Attention
     ↓
Residual Add
     ↓
LayerNorm
     ↓
FFN
     ↓
Residual Add
     ↓
LayerNorm
```

The shortcut path exists around both:

* Multi-Head Attention
* Feed Forward Network

---

## Why Residual Connections Matter

Without them:

* Deep transformers become difficult to optimize
* Training becomes unstable
* Performance degrades

With them:

* Better gradient flow
* Easier optimization
* Deeper networks become possible

---

## Key Takeaways

* Residual connections add the input back to the output.
* They help gradients flow through deep networks.
* They reduce optimization difficulties.
* They are essential for training modern transformers.
