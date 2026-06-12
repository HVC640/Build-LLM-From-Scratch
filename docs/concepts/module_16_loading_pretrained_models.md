# Module 16: Saving, Loading and Using Pretrained Models

## Why Save A Model?

Training GPT is expensive.

Even GPT-2 Small requires significant computation.

After training completes:

```text id="mjlwm33"
Weights
```

contain everything the model has learned.

These weights must be saved.

---

## What Is Saved?

Typically:

```text id="mjlwm34"
Model Parameters

Optimizer State

Training Metadata
```

are stored.

The most important component is:

```text id="mjlwm35"
State Dictionary
```

which contains all learned weights.

---

## Saving A Model

Purpose:

```text id="mjlwm36"
Persist Learned Knowledge
```

so training does not need to start from scratch.

---

## Loading A Model

Loading restores:

```text id="mjlwm37"
Previously Learned Parameters
```

into the architecture.

The architecture must match the saved model.

---

## Why Load Pretrained Models?

Training from scratch is expensive.

Instead:

```text id="mjlwm38"
Use Existing Foundation Model
```

and build on top of it.

Examples:

```text id="mjlwm39"
GPT-2

Llama

Mistral

Gemma
```

---

## Transfer Learning

A pretrained model already understands:

* Language
* Grammar
* Facts
* Syntax

We can then:

```text id="mjlwm40"
Fine Tune
```

for specific tasks.

---

## Example Workflow

```text id="mjlwm41"
Pretraining
      ↓

Save Weights
      ↓

Load Weights
      ↓

Fine Tune
      ↓

Deploy
```

This is how most modern AI systems are built.

---

## Why This Matters

Nobody trains GPT-sized models from scratch for every application.

Most real-world GenAI projects rely on:

```text id="mjlwm42"
Pretrained Foundation Models
```

followed by:

```text id="mjlwm43"
Fine-Tuning
or
RAG
```

---

## Key Takeaways

* Training produces learned weights.
* Saving preserves those weights.
* Loading restores learned knowledge.
* Pretrained models enable transfer learning.
* Most production AI systems start from pretrained models rather than training from scratch.
