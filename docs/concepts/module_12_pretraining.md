# Module 12: GPT Pretraining

## What is Pretraining?

Pretraining is the process of teaching a GPT model the statistical structure of language.

The model is exposed to massive amounts of text and repeatedly performs one task:

```text id="k2ubx9"
Predict the next token.
```

Everything GPT learns during pretraining emerges from this objective.

---

## Why Pretraining Matters

Before pretraining:

```text id="s7fdpc"
Random Weights
```

The model has no understanding of:

* Grammar
* Facts
* Language Structure
* Reasoning Patterns
* Writing Styles

Its predictions are essentially random.

---

After pretraining:

```text id="z04ljx"
Learned Weights
```

The model develops:

* Language understanding
* Semantic relationships
* World knowledge
* Pattern recognition

All from next-token prediction.

---

## The Training Data

Example text:

```text id="1b1vuy"
Every effort moves you forward.
```

Tokenized:

```text id="s6v3iu"
[1, 2, 3, 4, 5]
```

The sequence is converted into:

Input:

```text id="pt5b6j"
[1, 2, 3, 4]
```

Target:

```text id="6vbqca"
[2, 3, 4, 5]
```

---

## The Learning Objective

The model learns:

```text id="mb7s4d"
Every               → effort

Every effort        → moves

Every effort moves  → you

Every effort moves you → forward
```

All predictions are made simultaneously in a single forward pass.

---

## Why This Is Powerful

A simple objective:

```text id="lq4mb6"
Predict Next Token
```

forces the model to learn:

* Syntax
* Grammar
* Semantics
* Facts
* Long-range dependencies

because all of these help reduce prediction error.

---

## What Does GPT Actually Learn?

GPT does not explicitly learn:

```text id="1hzqsa"
Grammar Rules
Physics
History
Programming
```

Instead it learns:

```text id="p6kh95"
Statistical Patterns
```

present in the training data.

Knowledge emerges because predicting text requires understanding those patterns.

---

## The Objective Function

The model outputs:

```text id="g05mra"
Vocabulary Probabilities
```

for every position.

Cross Entropy compares:

```text id="hm34vt"
Predicted Distribution
```

with

```text id="5yjlwm"
Correct Token
```

and produces a loss value.

---

## Learning Process

Repeatedly:

```text id="c6z4nl"
Predict
↓
Measure Error
↓
Update Weights
↓
Predict Again
```

Eventually:

```text id="wmjlwm"
Loss Decreases
```

and

```text id="sru6qq"
Predictions Improve
```

---

## What Emerges From Pretraining?

Surprisingly, many abilities emerge naturally:

* Text completion
* Summarization
* Translation
* Coding assistance
* Reasoning patterns

These capabilities are not explicitly programmed.

They emerge from learning language distributions.

---

## Mental Model

Imagine giving a student billions of fill-in-the-blank exercises.

Example:

```text id="qqagfi"
The capital of France is _____
```

After enough practice, the student naturally acquires knowledge.

GPT learns in a similar way.

---

## Key Takeaways

* Pretraining teaches GPT through next-token prediction.
* Massive text corpora are used as training data.
* The objective is to minimize Cross Entropy Loss.
* Grammar, facts, and reasoning emerge indirectly.
* The model repeatedly predicts, measures error, and updates weights.
* Pretraining creates a foundation model.
