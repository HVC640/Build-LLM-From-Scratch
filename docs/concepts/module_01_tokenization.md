# Module 1: Tokenization & Data Preparation

## What is Tokenization?

LLMs cannot understand raw text.

Text must first be converted into numerical representations called tokens.

Example:

```text
I love AI
```

might become:

```text
[40, 1842, 9552]
```

The process of converting text into tokens is called tokenization.

---

## Why Do We Need Tokenization?

Neural networks operate on numbers.

They cannot process:

```text
hello
world
GPT
```

directly.

Tokenization acts as a bridge between human language and numerical computation.

---

## Types of Tokenization

### Character-Level Tokenization

Example:

```text
hello
```

becomes:

```text
[h, e, l, l, o]
```

#### Advantages

* Small vocabulary
* No unknown words
* Simple implementation

#### Disadvantages

* Long sequences
* Slow training
* Difficult to learn language patterns

---

### Word-Level Tokenization

Example:

```text
I love machine learning
```

becomes:

```text
[I, love, machine, learning]
```

#### Advantages

* Short sequences
* Easier semantic understanding

#### Disadvantages

* Huge vocabulary
* Out-of-vocabulary problem
* Memory intensive

---

### Subword Tokenization

Example:

```text
unbelievable
```

might become:

```text
un
believ
able
```

This combines the strengths of character and word tokenization.

---

## Byte Pair Encoding (BPE)

BPE is the tokenization strategy used by GPT-2.

### Core Idea

Start with characters.

Repeatedly merge the most common adjacent character pairs.

Example:

```text
l o w
l o w e r
```

Frequent pairs:

```text
l + o -> lo
lo + w -> low
```

Over time, common words become single tokens while rare words remain decomposable.

---

## Why GPT Uses BPE

BPE balances:

* Vocabulary size
* Sequence length
* Generalization

Compared to word-level tokenization:

* Smaller vocabulary

Compared to character-level tokenization:

* Shorter sequences

---

## Out-of-Vocabulary Problem

Word tokenizers fail on unseen words.

Example:

```text
electroencephalographic
```

Subword tokenizers can break it into known pieces.

This eliminates the OOV problem.

---

## Context Window

An LLM processes tokens, not words.

Example:

```text
Context Window = 1024 tokens
```

The model can only attend to the most recent 1024 tokens.

This concept becomes important later when studying attention.

---

## Key Takeaways

* LLMs operate on tokens, not text.
* Tokenization converts language into numbers.
* Character tokenization creates long sequences.
* Word tokenization creates huge vocabularies.
* BPE provides a practical balance.
* GPT-2 uses Byte Pair Encoding.
* Context windows are measured in tokens.
