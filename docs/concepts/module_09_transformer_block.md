# Module 9: Transformer Block (GPT Block)

## The Big Picture

By itself, attention is not a complete neural network.

A GPT model is built by repeatedly stacking a component called a Transformer Block.

A Transformer Block combines:

* Multi-Head Attention
* Layer Normalization
* Feed Forward Network (FFN)
* Residual Connections

into a single reusable unit.

---

## Why Do We Need a Transformer Block?

Attention is excellent at answering:

```text
Which tokens are important?
```

However, attention alone cannot perform sufficient computation.

We also need:

```text
Processing
Reasoning
Feature Extraction
Knowledge Storage
```

This is provided by the Feed Forward Network.

The Transformer Block combines:

```text
Communication + Computation
```

into one architecture.

---

## GPT Block Architecture

A GPT Block follows this structure:

```text
Input
  │
  ▼

LayerNorm
  │
  ▼

Multi-Head Attention
  │
  ▼

Residual Add
  │
  ▼

LayerNorm
  │
  ▼

Feed Forward Network
  │
  ▼

Residual Add
  │
  ▼

Output
```

---

## Step 1: Layer Normalization

Input embeddings enter the block.

Before attention is applied:

```text
x
 ↓
LayerNorm
```

LayerNorm stabilizes activations and improves training.

This approach is called:

```text
Pre-Norm Transformer
```

and is used in modern GPT models.

---

## Step 2: Multi-Head Attention

Attention receives normalized inputs.

It computes:

```text
Q = XWq
K = XWk
V = XWv
```

Then:

```text
Attention(Q,K,V)
```

for multiple heads simultaneously.

Purpose:

```text
Allow tokens to communicate.
```

Example:

```text
"The animal didn't cross the street because it was tired."
```

Attention helps determine:

```text
it → animal
```

---

## Step 3: First Residual Connection

After attention:

```text
Attention Output
```

is added back to the original input.

```text
Output = x + Attention(x)
```

Why?

Because:

```text
Attention should enhance information,
not replace it.
```

Residual connections also improve gradient flow.

---

## Step 4: Second LayerNorm

The attention-enhanced representation is normalized again.

```text
x
 ↓
LayerNorm
```

This keeps activations stable before entering the FFN.

---

## Step 5: Feed Forward Network

The FFN performs actual computation.

Structure:

```text
Linear
 ↓
GELU
 ↓
Linear
```

Example for GPT-2 Small:

```text
768
 ↓
3072
 ↓
768
```

Purpose:

```text
Transform information gathered by attention.
```

---

## Step 6: Second Residual Connection

The FFN output is added back.

```text
Output = x + FFN(x)
```

Again:

```text
Keep original information
+
Add learned improvements
```

---

## Intuition

Think of a Transformer Block as a meeting room.

### Attention

Everyone shares information.

```text
Who should I listen to?
```

---

### FFN

Each person privately thinks about what they heard.

```text
What does this information mean?
```

---

### Residual Connections

Nobody forgets what they knew before entering the meeting.

---

### LayerNorm

Ensures everyone communicates using the same scale and format.

---

## Communication vs Computation

A useful mental model:

### Attention

```text
Communication
```

Moves information between tokens.

---

### FFN

```text
Computation
```

Processes information within each token.

---

## Important Observation

Attention mixes information across tokens.

Example:

```text
Token A learns from Token B.
```

FFN does NOT.

The FFN operates independently on each token.

Example:

```text
Token A processed separately
Token B processed separately
Token C processed separately
```

This is a very common interview question.

---

## Why Stack Multiple Blocks?

A single block can only perform limited reasoning.

GPT stacks many blocks.

Example:

### GPT-2 Small

```text
12 Transformer Blocks
```

### GPT-2 Medium

```text
24 Transformer Blocks
```

### GPT-2 Large

```text
36 Transformer Blocks
```

### GPT-2 XL

```text
48 Transformer Blocks
```

Each block gradually refines token representations.

---

## What Happens Across Layers?

Layer 1:

```text
Basic relationships
```

Layer 4:

```text
Grammar
```

Layer 8:

```text
Semantics
```

Layer 12:

```text
Higher-level concepts
```

The representation becomes increasingly sophisticated as it moves upward.

---

## Transformer Block Formula

A GPT Block can be summarized as:

```text
x = x + Attention(LayerNorm(x))

x = x + FFN(LayerNorm(x))
```

This tiny formula describes the core computation repeated throughout GPT.

---

## Key Takeaways

* A Transformer Block is the fundamental building block of GPT.
* It combines Attention, FFN, LayerNorm, and Residual Connections.
* Attention enables communication between tokens.
* FFN performs computation on each token.
* Residual connections preserve information and improve gradient flow.
* LayerNorm stabilizes training.
* GPT is created by stacking many Transformer Blocks.
