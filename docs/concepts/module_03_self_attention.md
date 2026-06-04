# Module 3: Self-Attention

## The Problem

Traditional sequence models such as RNNs process tokens one by one.

As sequences become longer, it becomes difficult to retain information from earlier tokens.

Example:

```text
The cat that sat on the mat was hungry.
```

To understand the word "hungry", the model may need information from tokens much earlier in the sequence.

Self-attention was introduced to allow every token to directly access information from every other token.

---

## Simplified Attention Mechanism

The goal of attention is to determine:

> Which tokens are important for understanding the current token?

The process consists of three steps.

### Step 1: Compute Attention Scores

Measure similarity between the current token and every other token.

Higher score means higher relevance.

---

### Step 2: Compute Attention Weights

Apply SoftMax to attention scores.

```text
Attention Scores
        ↓
      SoftMax
        ↓
Attention Weights
```

The resulting values:

* Sum to 1
* Can be interpreted as probabilities
* Indicate relative importance

---

### Why SoftMax?

SoftMax converts arbitrary scores into a probability distribution.

Benefits:

* Produces values between 0 and 1
* All values sum to 1
* Makes importance comparison easier
* Uses exponentiation to stabilize optimization

Example:

```text
[2.0, 1.0, 0.1]
```

becomes

```text
[0.66, 0.24, 0.10]
```

---

### Step 3: Compute Context Vector

The context vector is a weighted combination of all input vectors.

```text
Context Vector
=
Attention Weights × Input Vectors
```

The context vector contains information gathered from the entire sequence.

---

## Limitation of Simplified Attention

The attention calculation depends directly on the input vectors.

There are no trainable parameters.

As a result:

* Limited flexibility
* Cannot learn sophisticated relationships
* Cannot adapt attention patterns during training

This motivates trainable self-attention.

---

## Self-Attention with Trainable Weights

Self-attention introduces three trainable projections:

* Query (Wq)
* Key (Wk)
* Value (Wv)

These are learned during training.

---

## Query, Key and Value Intuition

### Query

Represents:

> What information am I looking for?

Analogous to a search query in a search engine.

---

### Key

Represents:

> What information does this token contain?

The query is compared against keys.

---

### Value

Represents:

> The actual information to be retrieved.

Once relevance is determined using Query and Key, the corresponding Values are aggregated.

---

## Attention Pipeline

### Step 1: Create Q, K and V

```text
Q = XWq
K = XWk
V = XWv
```

---

### Step 2: Compute Attention Scores

```text
Attention Scores
=
Q × Kᵀ
```

Each token compares itself against every other token.

---

### Step 3: Scale Scores

```text
(Q × Kᵀ) / √d_k
```

where:

```text
d_k = key dimension
```

---

### Why Scale by √d_k ?

Without scaling:

* Dot products become very large
* SoftMax becomes extremely peaky
* Gradients become tiny
* Learning becomes unstable

Scaling keeps variance approximately constant and improves optimization.

---

### Step 4: Compute Attention Weights

```text
SoftMax(
(Q × Kᵀ) / √d_k
)
```

---

### Step 5: Compute Context Vectors

```text
Attention Weights × V
```

This produces the final contextual representation.

---

## Why Trainable Weights Matter

The matrices:

```text
Wq
Wk
Wv
```

are learned through gradient descent.

This allows the model to:

* Learn syntactic relationships
* Learn semantic relationships
* Capture long-range dependencies
* Adapt attention patterns to the task

Without trainable weights, attention remains a fixed similarity mechanism.

With trainable weights, attention becomes a learned retrieval system.

---

## Computational Complexity

For a sequence length n:

```text
O(n²)
```

Every token attends to every other token.

This quadratic complexity is one of the biggest limitations of Transformer architectures.

---

## Key Takeaways

* Attention determines which tokens matter.
* SoftMax converts scores into probabilities.
* Context vectors are weighted combinations of values.
* Query asks, Key identifies, Value provides information.
* Self-attention introduces trainable projections.
* Scaling by √d_k improves optimization stability.
* Self-attention allows direct interaction between all tokens.
