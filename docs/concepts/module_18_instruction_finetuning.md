# Module 18: Instruction Fine-Tuning

## What is Instruction Fine-Tuning?

Instruction Fine-Tuning (IFT) teaches a pretrained GPT model to follow human instructions.

Example:

```text
User:
Write a Python function.

Assistant:
def hello():
    ...
```

Unlike classification:

```text
Output = Label
```

Instruction tuning keeps:

```text
Output = Text
```

---

# Why Is It Needed?

A pretrained GPT learns:

```text
Predict Next Token
```

but does not necessarily learn:

```text
Follow Instructions
```

Instruction tuning aligns the model with human tasks.

---

# Dataset Format

Training examples usually contain:

```text
Instruction

Input (optional)

Response
```

Example:

```text
Instruction:
Translate to French

Input:
Hello

Response:
Bonjour
```

---

# Stage 1: Dataset Preparation

## Download Dataset

Instruction-response pairs.

---

## Format Dataset

Convert conversations into text sequences.

---

## Create DataLoaders

Batch examples for training.

---

# Stage 2: Fine-Tuning

## Load Pretrained Model

Start from GPT weights.

---

## Train On Instruction Data

The model learns:

```text
Instruction
↓
Desired Response
```

patterns.

---

# Why Use -100?

During batching:

Sequences often have different lengths.

Example:

```text
Response A
```

may be shorter than:

```text
Response B
```

Padding is added.

---

Problem:

```text
Padding Tokens
```

are not real targets.

We do not want them contributing to loss.

---

Solution:

```text
Target = -100
```

PyTorch CrossEntropyLoss ignores these positions.

This prevents padding from affecting training.

---

# Important Observation

Instruction tuning usually keeps:

```text
Same GPT Architecture
```

No classifier head is added.

The model still predicts:

```text
Next Token
```

The training data simply changes.

---

# Evaluation

Evaluation often includes:

## Qualitative Evaluation

Human inspection.

Example:

```text
Is the answer helpful?
```

---

## Quantitative Evaluation

Scores:

```text
BLEU

ROUGE

Exact Match

LLM-as-a-Judge
```

depending on the task.

---

# Key Takeaways

* Instruction tuning teaches GPT to follow instructions.
* The architecture usually remains unchanged.
* The next-token objective remains unchanged.
* Training data changes from raw text to instruction-response pairs.
* Padding tokens are commonly masked using -100.
