# Module 11: Cross Entropy Loss and Perplexity

## The Core Problem

GPT learns one task:

```text
Predict the next token.
```

Suppose the input is:

```text
Every effort moves you
```

and the correct next token is:

```text
forward
```

The model outputs probabilities for every token in the vocabulary.

Example:

```text
forward      -> 0.70
backward     -> 0.10
quickly      -> 0.05
tomorrow     -> 0.03
...
```

Now we need a way to answer:

```text
How good was this prediction?
```

This is the purpose of a loss function.

---

# What Makes a Good Loss Function?

A good loss function should:

1. Reward correct predictions.
2. Penalize incorrect predictions.
3. Penalize confident mistakes heavily.
4. Be differentiable.
5. Produce useful gradients for optimization.

Cross Entropy satisfies all these requirements.

---

# Why Not Use Accuracy?

A common beginner question:

```text
Why not simply measure accuracy?
```

Example:

Prediction A:

```text
forward -> 0.51
```

Prediction B:

```text
forward -> 0.99
```

Both are:

```text
Correct
```

Accuracy treats them equally.

However:

```text
0.99
```

is much better than:

```text
0.51
```

Cross Entropy captures this difference.

---

# Intuition Behind Cross Entropy

Imagine an exam.

Correct answer:

```text
forward
```

---

Prediction 1:

```text
forward -> 0.99
```

The model is:

```text
Very confident
and
Correct
```

Loss should be very small.

---

Prediction 2:

```text
forward -> 0.50
```

The model is:

```text
Unsure
```

Loss should be larger.

---

Prediction 3:

```text
forward -> 0.01
```

The model is:

```text
Very confident
and
Wrong
```

Loss should be huge.

Cross Entropy behaves exactly this way.

---

# The Key Idea

Cross Entropy focuses only on:

```text
Probability assigned to the correct token.
```

If the correct token receives:

```text
High Probability
```

loss decreases.

If the correct token receives:

```text
Low Probability
```

loss increases.

---

# Why Use Logarithms?

This is one of the most important insights.

Suppose:

```text
P(correct) = 0.99
```

and

```text
P(correct) = 0.01
```

The difference is large.

But when training LLMs:

```text
Millions of probabilities
```

are multiplied together.

---

Example:

```text
0.9 × 0.8 × 0.7 × 0.95 × ...
```

After many multiplications:

```text
≈ 0
```

due to numerical underflow.

---

## Solution

Use logarithms.

Logarithms convert:

```text
Multiplication
```

into

```text
Addition
```

which is much more stable.

Example:

```text
log(a × b)

=
log(a) + log(b)
```

---

# Negative Log Likelihood

Cross Entropy is essentially:

```text
Take probability of correct token
↓
Apply log
↓
Negate it
```

High probability:

```text
Small Loss
```

Low probability:

```text
Large Loss
```

This gives us a smooth optimization objective.

---

# Training Objective

The goal of GPT training is:

```text
Maximize Probability
of Correct Tokens
```

Since optimizers minimize functions:

```text
Maximize Probability
```

becomes:

```text
Minimize Cross Entropy Loss
```

These are mathematically equivalent objectives.

---

# Example

Suppose:

```text
Correct Token = forward
```

Case 1:

```text
P(forward) = 0.9
```

Loss:

```text
Small
```

---

Case 2:

```text
P(forward) = 0.1
```

Loss:

```text
Large
```

The model receives gradients that push probability mass toward the correct token.

---

# What Is Perplexity?

Perplexity is derived directly from Cross Entropy.

It is simply a more interpretable metric.

---

## Intuition

Imagine a vocabulary of:

```text
50,000 tokens
```

Suppose the model is completely clueless.

Then every token gets:

```text
1 / 50000
```

probability.

The model is essentially thinking:

```text
Could be any of 50,000 words.
I have no idea.
```

Perplexity measures this uncertainty.

---

# What Perplexity Represents

Perplexity can be interpreted as:

```text
How many equally likely choices
the model is considering.
```

---

Example:

Perplexity:

```text
50000
```

means:

```text
I am completely confused.
```

---

Perplexity:

```text
100
```

means:

```text
I have narrowed it down
to roughly 100 possibilities.
```

---

Perplexity:

```text
10
```

means:

```text
I am much more confident.
```

---

Perplexity:

```text
1
```

means:

```text
Perfect prediction.
```

---

# Relationship Between Loss and Perplexity

Perplexity is simply:

Perplexity=e^{CrossEntropyLoss}

Because of this:

```text
Low Loss
↓
Low Perplexity
```

and

```text
High Loss
↓
High Perplexity
```

They measure the same thing from different perspectives.

---

# Why Report Perplexity?

Cross Entropy:

```text
Great for optimization.
```

Perplexity:

```text
Great for interpretation.
```

Researchers often report:

```text
Validation Perplexity
```

because it is easier to understand.

---

# What GPT Actually Optimizes

GPT does NOT optimize:

```text
Accuracy
BLEU
ROUGE
Human Preference
```

during pretraining.

GPT optimizes:

```text
Cross Entropy Loss
```

which is equivalent to maximizing:

```text
Probability of the next token.
```

Everything else emerges from this objective.

---

# Mental Model

Imagine GPT taking a multiple-choice exam.

Cross Entropy asks:

```text
How much probability
did you assign
to the correct answer?
```

Perplexity asks:

```text
How many answers
did you seem uncertain between?
```

Both measure prediction quality.

One is used for optimization.

The other is used for interpretation.

---

# Key Takeaways

* GPT is trained using Cross Entropy Loss.
* Cross Entropy rewards assigning high probability to correct tokens.
* Confident mistakes are penalized heavily.
* Logarithms provide numerical stability and useful gradients.
* Minimizing Cross Entropy is equivalent to maximizing likelihood.
* Perplexity is an interpretable transformation of Cross Entropy.
* Lower Loss and Lower Perplexity both indicate a better language model.
