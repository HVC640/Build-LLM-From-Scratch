# Generation vs Training

## The Question

When generating text, how does GPT know:

```text
How many tokens to generate?
```

For example:

```text
Input:
Every effort moves you
```

How does GPT decide whether to generate:

```text
forward
```

or

```text
forward in life if you continue to remain consistent and disciplined...
```

Where does generation stop?

---

# The Key Insight

GPT learns only one task:

```text
Predict the next token.
```

It does NOT learn:

```text
How many tokens to generate.
```

The stopping behavior is controlled by the generation algorithm.

---

# Training vs Inference

This distinction is extremely important.

---

## During Training

Suppose the sequence is:

```text
Every effort moves you forward
```

Tokenized:

```text
[1, 2, 3, 4, 5]
```

Input:

```text
[1, 2, 3, 4]
```

Target:

```text
[2, 3, 4, 5]
```

The model predicts:

```text
Position 1 → Token 2

Position 2 → Token 3

Position 3 → Token 4

Position 4 → Token 5
```

simultaneously.

---

## Important Observation

GPT does NOT produce one prediction.

It produces:

```text
One prediction per token position.
```

For a sequence length of:

```text
256
```

the model produces:

```text
256 next-token predictions
```

in a single forward pass.

This is what makes GPT training efficient.

---

# During Inference

Suppose the input is:

```text
Every effort moves you
```

The model produces logits for every position.

However, we only care about:

```text
The final token position.
```

which corresponds to:

```text
you
```

because that token contains the richest context.

---

## Why The Last Token?

The representation of:

```text
you
```

has attended to:

```text
Every
effort
moves
you
```

through multiple Transformer layers.

Its final embedding contains information from the entire sequence.

Therefore:

```text
Logits("you")
```

are used to predict:

```text
forward
```

---

# Autoregressive Generation

After predicting:

```text
forward
```

the sequence becomes:

```text
Every effort moves you forward
```

The new sequence is fed back into the model.

Now the model predicts:

```text
the next token after forward
```

This process repeats.

---

## Generation Loop

Conceptually:

```text
Predict Next Token
        ↓

Append Token
        ↓

Predict Next Token
        ↓

Append Token
        ↓

Repeat
```

This is called:

```text
Autoregressive Generation
```

---

# How Does Generation Stop?

GPT itself does not decide.

The generation algorithm decides.

---

## Method 1: Fixed Token Limit

Example:

```python
max_new_tokens = 50
```

Generation stops after:

```text
50 new tokens
```

have been produced.

---

## Method 2: End Of Sequence Token

Most language models are trained using a special token:

```text
<EOS>
```

meaning:

```text
End Of Sequence
```

Example:

```text
Every effort moves you forward <EOS>
```

The model learns:

```text
forward → <EOS>
```

When:

```text
<EOS>
```

is generated, text generation stops.

---

## Method 3: Context Window Limit

Every GPT model has a maximum context length.

Examples:

```text
GPT-2 Small → 1024 Tokens

GPT-3 → 2048 Tokens

Modern LLMs → 8K, 32K, 128K+
```

Once the context window is reached:

```text
Generation must stop
```

or

```text
Old tokens must be removed
```

depending on the implementation.

---

# What GPT Actually Learns

GPT does not learn:

```text
Sentence Structure

Document Length

When To Stop
```

directly.

GPT learns only:

```text
P(next_token | previous_tokens)
```

which means:

```text
Probability of the next token
given all previous tokens.
```

Everything else is built on top of this capability.

---

# Mental Model

Training:

```text
Predict all next tokens
at all positions
simultaneously.
```

Inference:

```text
Use only the final token's prediction,
append the predicted token,
and repeat.
```

---

# One Sentence Summary

During training, GPT predicts the next token for every position in parallel, while during inference it repeatedly uses the final token's context-aware representation to generate one new token at a time until a stopping condition is reached.
