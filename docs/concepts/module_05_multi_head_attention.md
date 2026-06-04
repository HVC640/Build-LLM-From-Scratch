# Module 5: Multi-Head Attention

## Motivation

A single attention mechanism can only learn one type of relationship at a time.

However language contains many relationships simultaneously:

* Grammar
* Syntax
* Semantics
* Coreference
* Long-range dependencies

A single attention head may struggle to capture all of them.

---

## What is Multi-Head Attention?

Instead of using one attention operation:

```text
Attention(Q, K, V)
```

we create multiple independent attention heads.

Example:

```text
Head 1
Head 2
Head 3
Head 4
...
```

Each head learns different attention patterns.

---

## How It Works

### Step 1

Project input into:

```text
Q1 K1 V1
Q2 K2 V2
Q3 K3 V3
...
```

Each head has its own learned parameters.

---

### Step 2

Perform causal self-attention independently.

```text
Head 1 Attention
Head 2 Attention
Head 3 Attention
...
```

---

### Step 3

Concatenate outputs.

```text
Head1 || Head2 || Head3 || ...
```

---

### Step 4

Apply final projection matrix.

```text
Wo
```

This combines information from all heads.

---

## Intuition

Each head acts as a different perspective.

Examples:

Head A:

```text
Subject-Verb relationships
```

Head B:

```text
Long-distance dependencies
```

Head C:

```text
Semantic meaning
```

Head D:

```text
Position-related information
```

The model learns these automatically during training.

---

## Why Not One Large Head?

Using multiple heads allows:

* Multiple representation subspaces
* Parallel pattern discovery
* Better expressiveness

A single head tends to collapse information into one representation.

---

## Benefits

### Richer Representations

Multiple perspectives are learned simultaneously.

### Better Long-Range Reasoning

Different heads specialize in different dependency lengths.

### Improved Learning Capacity

The model can represent more complex relationships.

---

## Key Takeaways

* Multi-head attention is multiple attention mechanisms running in parallel.
* Each head has independent Q, K and V projections.
* Outputs are concatenated and projected.
* Different heads learn different linguistic patterns.
* Multi-head attention significantly increases model expressiveness.
