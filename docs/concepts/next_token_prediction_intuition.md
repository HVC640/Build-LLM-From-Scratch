# Next Token Prediction Intuition

## The Most Important GPT Insight

When GPT receives:

```text id="l6w1hb"
Every effort moves you
```

many beginners imagine:

```text id="y9w06v"
Sentence
    ↓
Model
    ↓
Next Word
```

This mental model is incomplete.

GPT actually predicts the next token for every position simultaneously.

---

## What Does GPT Output?

Suppose input tokens are:

```text id="9ixr2o"
Every
effort
moves
you
```

The model produces:

```text id="rr7d0m"
Logits for Every
Logits for effort
Logits for moves
Logits for you
```

Each token position gets its own vocabulary prediction.

---

## Why Does This Work?

Because of causal self-attention.

Each token can only see:

* Previous tokens
* Current token

and cannot see future tokens.

Therefore:

```text id="cl4lqo"
Every
```

only knows:

```text id="cn7cga"
Every
```

---

```text id="8h9c8d"
effort
```

knows:

```text id="tqhlit"
Every effort
```

---

```text id="bhm2v5"
moves
```

knows:

```text id="hf0l76"
Every effort moves
```

---

```text id="6t4c43"
you
```

knows:

```text id="x8l5d3"
Every effort moves you
```

---

## Why Use the Last Token's Logits?

Suppose we want:

```text id="8rqgjk"
Every effort moves you
```

to predict:

```text id="h15mbt"
forward
```

We examine the logits produced at:

```text id="cw3kzt"
you
```

because that representation contains the most context.

The token "you" has attended to:

```text id="1avm7s"
Every
effort
moves
you
```

through multiple Transformer layers.

Its final representation summarizes the entire sequence.

---

## What Does the "You" Vector Actually Contain?

After passing through all Transformer Blocks:

```text id="l1lq6o"
you
```

is no longer merely the embedding of the word "you".

It becomes:

```text id="pkqm8r"
Contextualized Representation
```

containing information from:

```text id="g1k6cf"
Every
effort
moves
you
```

This contextualized representation is what the output head uses to predict:

```text id="f3t3be"
forward
```

---

## Why This Matters

The model is not predicting based on:

```text id="s4o7pb"
you
```

alone.

It is predicting based on:

```text id="6s9jlwm"
Everything it knows up to "you".
```

This is the core idea behind autoregressive language modeling.

---

## Training Insight

During training:

Input:

```text id="gvd8z4"
Every effort moves you
```

Targets:

```text id="d4d2vk"
effort
moves
you
forward
```

The model learns:

```text id="4lz1ig"
Every  → effort

Every effort → moves

Every effort moves → you

Every effort moves you → forward
```

all in a single forward pass.

This is why GPT training is efficient.

---

## Mental Model

Think of each token position as asking:

```text id="jlwmcm"
Based on everything I have seen so far,
what comes next?
```

The further right a token is, the more context it possesses.

The last token therefore contains the richest context for predicting the next token after the sequence.

---

## One Sentence Summary

The logits corresponding to the final token are used for next-token generation because that token's representation has been enriched by attention over the entire preceding context, making it the best representation for predicting what comes next.
