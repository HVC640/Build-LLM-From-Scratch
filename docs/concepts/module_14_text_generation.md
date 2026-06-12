# Module 14: Text Generation

## What Happens After Training?

After pretraining completes:

```text id="h8zzcm"
Input Text
      ↓

GPT Model
      ↓

Vocabulary Logits
```

The model does not directly output words.

Instead it outputs:

```text id="mg76xg"
Logits
```

for every token in the vocabulary.

---

## What Are Logits?

Suppose GPT sees:

```text id="6eblfk"
Every effort moves you
```

The output might be:

```text id="3gq4x7"
forward  → 8.7

backward → 3.1

slowly   → 1.2

today    → 0.4
```

These values are called logits.

They are not probabilities.

---

## Converting Logits to Probabilities

SoftMax transforms logits into probabilities.

Example:

```text id="l1fdtm"
forward  → 0.72

backward → 0.15

slowly   → 0.08

today    → 0.05
```

Now all values:

* Are between 0 and 1
* Sum to 1

---

## Selecting The Next Token

After probabilities are produced, a token selection strategy is required.

Common approaches:

```text id="xaqd4y"
Greedy Decoding

Temperature Sampling

Top-K Sampling

Top-P Sampling
```

---

## Autoregressive Generation

After selecting the next token:

```text id="0btrhp"
forward
```

the sequence becomes:

```text id="jg4o13"
Every effort moves you forward
```

This new sequence is fed back into GPT.

Generation repeats until:

* EOS token appears
* Context limit reached
* max_new_tokens reached

---

## Key Takeaways

* GPT outputs logits, not words.
* SoftMax converts logits into probabilities.
* Sampling strategies choose the next token.
* Generated tokens are repeatedly fed back into the model.
* This process is called autoregressive generation.
