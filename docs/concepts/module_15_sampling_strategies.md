# Module 15: Temperature Scaling and Top-K Sampling

## The Problem

Suppose GPT predicts:

```text id="mjlwm1"
forward  → 0.70

backward → 0.15

slowly   → 0.10

today    → 0.05
```

Should we always select:

```text id="mjlwm2"
forward
```

?

If yes:

```text id="mjlwm3"
Generation becomes repetitive.
```

If no:

```text id="mjlwm4"
Generation becomes more diverse.
```

Sampling strategies control this tradeoff.

---

# Greedy Decoding

The simplest approach:

```text id="mjlwm5"
Choose highest probability token.
```

Example:

```text id="mjlwm6"
forward → 0.70
```

Always select:

```text id="mjlwm7"
forward
```

---

## Advantages

* Deterministic
* Consistent
* Simple

---

## Disadvantages

* Repetitive outputs
* Less creativity
* Can get stuck in loops

---

# Temperature Scaling

Temperature modifies the probability distribution before sampling.

Temperature is denoted by:

```text id="mjlwm8"
T
```

---

## Intuition

Temperature controls:

```text id="mjlwm9"
Confidence
```

of the model.

---

### Low Temperature

Example:

```text id="mjlwm10"
T = 0.1
```

Distribution becomes:

```text id="mjlwm11"
Very Sharp
```

The highest probability token dominates.

Behavior:

```text id="mjlwm12"
Less Creative
More Deterministic
```

---

### High Temperature

Example:

```text id="mjlwm13"
T = 2.0
```

Distribution becomes:

```text id="mjlwm14"
Flatter
```

Other tokens gain probability mass.

Behavior:

```text id="mjlwm15"
More Creative
More Random
```

---

### Temperature = 1

```text id="mjlwm16"
Original Distribution
```

No modification occurs.

---

# Why Temperature Works

Temperature adjusts logits before SoftMax.

Conceptually:

```text id="mjlwm17"
Low Temperature
↓
Confidence Increases

High Temperature
↓
Confidence Decreases
```

The model itself does not change.

Only the sampling behavior changes.

---

# Multinomial Sampling

After obtaining probabilities:

```text id="mjlwm18"
forward → 0.70

backward → 0.15

slowly → 0.10

today → 0.05
```

we sample according to these probabilities.

---

## Example

Imagine drawing from a weighted lottery.

Token:

```text id="mjlwm19"
forward
```

wins most often.

However:

```text id="mjlwm20"
backward
```

can occasionally be selected.

This introduces diversity.

---

# Top-K Sampling

Problem:

High temperature can occasionally select terrible tokens.

Example:

```text id="mjlwm21"
forward  → 0.40

backward → 0.30

slowly   → 0.15

banana   → 0.00001
```

We don't want:

```text id="mjlwm22"
banana
```

appearing.

---

## Solution

Keep only:

```text id="mjlwm23"
Top K Most Likely Tokens
```

Example:

```text id="mjlwm24"
K = 3
```

Retain:

```text id="mjlwm25"
forward

backward

slowly
```

Remove:

```text id="mjlwm26"
banana
```

and all lower-ranked tokens.

---

## Renormalization

After removing tokens:

```text id="mjlwm27"
Probabilities are normalized again.
```

The remaining probabilities sum to 1.

---

## Benefits

Top-K:

```text id="mjlwm28"
Maintains Diversity
```

while avoiding:

```text id="mjlwm29"
Extremely Unlikely Tokens
```

---

# Combining Temperature and Top-K

Modern LLMs often use:

```text id="mjlwm30"
Temperature
+
Top-K
```

together.

Temperature controls creativity.

Top-K prevents nonsense outputs.

---

# Mental Model

Temperature:

```text id="mjlwm31"
How adventurous should the model be?
```

Top-K:

```text id="mjlwm32"
How many candidate words are allowed?
```

---

# One Sentence Summary

Temperature controls how confident the probability distribution is, while Top-K restricts generation to only the K most likely tokens, balancing creativity and quality.
