# Module 10: GPT Architecture

## What is GPT?

GPT stands for:

```text id="h1s1xj"
Generative
Pre-trained
Transformer
```

GPT is a decoder-only Transformer architecture designed for autoregressive text generation.

The core objective is:

```text id="b0xxgf"
Predict the next token.
```

---

## GPT Architecture Overview

A GPT model consists of:

```text id="5r8l0k"
Input Tokens
      ↓

Token Embeddings
      ↓

Positional Embeddings
      ↓

N Transformer Blocks
      ↓

Final LayerNorm
      ↓

Linear Output Head
      ↓

Vocabulary Logits
```

---

## Step 1: Input Tokens

Example:

```text id="9wqxup"
Every effort moves you
```

Tokenizer converts text into token IDs.

```text id="kfd53d"
[6109, 3626, 6100, 345]
```

These IDs become the model input.

---

## Step 2: Token Embeddings

Each token ID is mapped to a dense vector.

Example:

```text id="y67g9i"
345
```

becomes:

```text id="pwsgmk"
[0.12, -0.44, ...]
```

For GPT-2 Small:

```text id="2q5tqn"
Embedding Dimension = 768
```

---

## Step 3: Positional Embeddings

Attention itself does not understand word order.

Therefore positional embeddings are added.

```text id="w8zc42"
Input Embedding
=
Token Embedding
+
Position Embedding
```

---

## Step 4: Transformer Blocks

The embeddings pass through multiple Transformer Blocks.

Example:

```text id="vr5gtl"
GPT-2 Small = 12 Blocks
```

Each block performs:

```text id="txw4vu"
LayerNorm
↓
Multi-Head Attention
↓
Residual Add
↓
LayerNorm
↓
FFN
↓
Residual Add
```

Each layer progressively enriches token representations.

---

## Step 5: Final LayerNorm

After the final Transformer Block:

```text id="4icxg5"
LayerNorm
```

is applied once more.

This stabilizes representations before prediction.

---

## Step 6: Output Projection Layer

A linear layer projects:

```text id="4z7y2o"
Embedding Dimension
```

to

```text id="wjlwm2"
Vocabulary Size
```

Example:

```text id="japkjv"
768 → 50257
```

for GPT-2.

---

## Step 7: Logits

Output:

```text id="tecf3v"
Batch
×
Sequence Length
×
Vocabulary Size
```

Example:

```text id="vyqukv"
[1, 4, 50257]
```

Each position receives a score for every vocabulary token.

These scores are called:

```text id="zltzk0"
Logits
```

---

## Important Insight

GPT does not produce:

```text id="a0pcnr"
One Prediction
```

It produces:

```text id="kydjvl"
One Prediction Per Token Position
```

simultaneously.

This is the foundation of efficient training.

---

## Key Takeaways

* GPT is a decoder-only Transformer.
* It predicts the next token.
* Token and positional embeddings create the input representation.
* Transformer Blocks enrich token representations.
* The output layer projects embeddings into vocabulary logits.
* GPT generates logits for every position simultaneously.
