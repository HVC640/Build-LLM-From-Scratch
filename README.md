# Build-LLM-From-Scratch

A hands-on implementation of a GPT-style Large Language Model (LLM) from scratch while documenting every major concept involved in modern Generative AI systems.

## Objective

The goal of this repository is **not** to train a state-of-the-art model.

Instead, the goal is to understand:

* How tokenization works
* How embeddings are learned
* How attention mechanisms operate
* How transformer architectures are built
* How pretraining and finetuning work
* How modern LLMs generate text

By the end of this project, I should be able to:

* Explain every major LLM component in interviews
* Understand the strengths and limitations of LLMs
* Debug and modify transformer architectures
* Connect LLM internals with real-world GenAI applications
* Build better RAG and AI systems

---

## Learning Roadmap

### Stage 1 - Building an LLM

#### Module 1 - Data Preparation & Tokenization

* Character Tokenization
* Word Tokenization
* Subword Tokenization
* Byte Pair Encoding (BPE)
* GPT Tokenizers
* Data Sampling

#### Module 2 - Embeddings

* One-Hot Encoding
* Embedding Layers
* Token Embeddings
* Positional Embeddings

#### Module 3 - Attention Mechanism

* Query, Key, Value
* Self Attention
* Causal Masking
* Scaled Dot Product Attention
* Multi Head Attention

#### Module 4 - LLM Architecture

* Feed Forward Networks
* Residual Connections
* Layer Normalization
* Transformer Blocks
* GPT Architecture

---

### Stage 2 - Foundation Model

#### Module 5 - Pretraining

* Next Token Prediction
* Language Modeling Objective

#### Module 6 - Training Loop

* Forward Pass
* Backpropagation
* AdamW
* Learning Rate Scheduling

#### Module 7 - Evaluation

* Validation Loss
* Perplexity
* Model Analysis

#### Module 8 - Loading Pretrained Weights

* GPT-2 Weights
* Transfer Learning

---

### Stage 3 - Finetuning

#### Module 9 - Classification Finetuning

* Sentiment Analysis
* Intent Detection
* Spam Detection

#### Module 10 - Instruction Finetuning

* Chat Models
* Instruction Following
* Personal Assistants

---

## Repository Structure

```text
Build-LLM-From-Scratch/
│
├── README.md
│
├── docs/
│   ├── concepts/
│   ├── interview-notes/
│   └── cheatsheets/
│
├── notebooks/
│
├── src/
│
├── experiments/
│
└── assets/
```

---

## Documentation Philosophy

Every topic contains:

### Notebook

Implementation and experiments.

### Concept Notes

Theory and explanations.

### Interview Notes

Common interview questions and answers.

---

## Why Learn LLM Internals?

Modern AI applications such as:

* RAG Systems
* AI Assistants
* Semantic Search
* Agentic Workflows
* Knowledge Retrieval Systems

all depend on understanding the underlying language model.

Learning how an LLM works internally makes it easier to:

* Design better prompts
* Improve retrieval pipelines
* Debug hallucinations
* Choose the correct architecture
* Understand model limitations

---

## Related Projects

### DocMind

A production-grade GenAI application focused on:

* RAG
* Advanced RAG
* Semantic Search
* Knowledge Retrieval

### ModelForge

A project focused on:

* Local LLM deployment
* Model serving
* Inference optimization

---

## Current Progress

* [x] Data Preparation & Sampling
* [ ] Embeddings
* [ ] Attention Mechanism
* [ ] Transformer Architecture
* [ ] Pretraining
* [ ] Training Loop
* [ ] Evaluation
* [ ] Loading Pretrained Weights
* [ ] Finetuning

---

## References

* Build a Large Language Model (Sebastian Raschka)
* GPT-2 Paper
* Attention Is All You Need
* Vizuara - Building LLM from Scratch Series

---

Building an LLM from scratch is one of the best ways to understand modern Generative AI systems beyond prompt engineering and APIs.
