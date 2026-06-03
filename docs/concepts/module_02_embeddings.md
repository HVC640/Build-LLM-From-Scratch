# Module 2: Embeddings

## The Problem

After tokenization, words become integer IDs.

Example:

```text
I love AI

[40, 1842, 9552]
```

The model still cannot understand meaning from these IDs.

For example:

```text
dog -> 100
cat -> 101
car -> 102
```

The numbers themselves contain no semantic information.

---

## One-Hot Encoding

One possible solution is one-hot encoding.

Example:

Vocabulary Size = 5

```text
dog -> [1,0,0,0,0]
cat -> [0,1,0,0,0]
car -> [0,0,1,0,0]
```

---

## Problems with One-Hot Encoding

### Sparse Vectors

Most values are zero.

Example:

```text
[0,0,0,0,1,0,0,0,0]
```

---

### No Semantic Meaning

The distance between:

```text
dog
cat
```

is identical to:

```text
dog
airplane
```

The representation contains no information about similarity.

---

### Memory Inefficient

For vocabularies with tens of thousands of tokens, vectors become huge.

---

## Embeddings

Embeddings solve this problem.

Instead of storing sparse vectors, we learn dense vectors.

Example:

```text
dog -> [0.12, -0.91, 0.44]
cat -> [0.09, -0.87, 0.47]
car -> [-0.74, 0.31, -0.15]
```

Notice:

```text
dog ≈ cat
```

while

```text
dog ≠ car
```

Semantically similar words develop similar embeddings.

---

## Embedding Layer

An embedding layer is essentially a lookup table.

Example:

Vocabulary Size:

```text
50000
```

Embedding Dimension:

```text
768
```

Shape:

```text
50000 x 768
```

Each row represents a token.

Each column represents a learned feature.

---

## Token Embeddings

Every token receives a dense vector.

Example:

```text
hello
```

becomes:

```text
[0.13, -0.22, ...]
```

These vectors are learned during training.

---

## Positional Embeddings

Attention alone does not understand order.

Example:

```text
dog bites man
```

vs

```text
man bites dog
```

contain the same words but different meanings.

Therefore GPT adds positional information.

---

## Input Representation in GPT

Final input:

```text
Input Embedding
=
Token Embedding
+
Positional Embedding
```

This combined representation is fed into the transformer.

---

## Why Embeddings Matter

Embeddings allow the model to:

* Learn semantic relationships
* Compress information efficiently
* Generalize across language
* Represent meaning numerically

Without embeddings, modern LLMs would not work.

---

## Key Takeaways

* Token IDs do not contain meaning.
* One-hot encoding is sparse and inefficient.
* Embeddings are dense learned representations.
* Similar words develop similar embeddings.
* GPT uses token embeddings and positional embeddings.
* Input to the transformer is the sum of both.
