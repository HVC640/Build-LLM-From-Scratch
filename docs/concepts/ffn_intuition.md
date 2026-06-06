# Feed Forward Network (FFN) Intuition

## The Biggest Misconception

Many people think:

```text
Attention = Thinking
FFN = Small Helper Layer
```

This is wrong.

A better mental model is:

```text
Attention = Communication
FFN = Computation
```

Attention gathers information.

FFN processes that information.

---

## What Does Attention Actually Do?

Attention answers:

```text
Which tokens are important?
```

Example:

```text
The animal didn't cross the street because it was tired.
```

Attention helps determine:

```text
"it" → "animal"
```

But attention itself does not deeply transform the representation.

It mainly moves information around.

---

## What Does FFN Do?

FFN answers:

```text
Now that I have gathered information,
what should I do with it?
```

This is where heavy computation happens.

---

## FFN Architecture

The FFN inside GPT is:

```text
Linear
 ↓
GELU
 ↓
Linear
```

---

## Dimension Expansion

Suppose:

```text
Embedding Dimension = 768
```

First layer expands:

```text
768 → 3072
```

This is:

```text
4 × Embedding Dimension
```

---

## Why Expand?

Think of it as:

```text
Temporary Workspace
```

The model creates a larger space where it can build richer intermediate representations.

Similar to:

```text
Taking notes on a large whiteboard
before writing the final answer.
```

---

## Why Shrink Back?

The second linear layer projects:

```text
3072 → 768
```

so the representation can continue through the transformer.

---

## Why Does FFN Have So Many Parameters?

For GPT-2 Small:

```text
Embedding Size = 768
```

FFN contains:

```text
768 × 3072
+
3072 × 768
=
4.7 Million Parameters
```

per transformer block.

Across 12 blocks:

```text
≈ 56.6 Million Parameters
```

Nearly half of GPT-2 Small's parameters.

---

## What Does FFN Learn?

FFNs often learn:

* Semantic concepts
* Linguistic patterns
* Entity information
* Factual associations
* Abstract features

Many researchers believe a large portion of the model's "knowledge" is stored in FFN weights.

---

## Attention vs FFN

Attention:

```text
Find relevant information
```

FFN:

```text
Process relevant information
```

Attention:

```text
Who should I listen to?
```

FFN:

```text
What should I think about what I heard?
```

Attention:

```text
Communication
```

FFN:

```text
Computation
```

---

## One Sentence Summary

Attention moves information between tokens, while the Feed Forward Network transforms that information into richer representations and performs much of the model's actual computation.
