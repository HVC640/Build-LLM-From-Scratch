# Pretraining vs Fine-Tuning

## The Most Important Distinction

Many beginners think:

```text
Pretraining
=
Fine-Tuning
```

They are not the same.

---

# Pretraining

Goal:

```text
Learn Language
```

Training Data:

```text
Books

Web Pages

Articles

Code
```

Objective:

```text
Predict Next Token
```

Result:

```text
Foundation Model
```

Example:

```text
GPT-2
GPT-3
Llama
Mistral
```

---

# Classification Fine-Tuning

Goal:

```text
Learn Specific Labels
```

Training Data:

```text
Text + Label
```

Objective:

```text
Predict Class
```

Result:

```text
Classifier
```

Example:

```text
Spam Detector

Sentiment Analyzer
```

---

# Instruction Fine-Tuning

Goal:

```text
Follow Human Instructions
```

Training Data:

```text
Instruction + Response
```

Objective:

```text
Generate Helpful Responses
```

Result:

```text
Assistant Model
```

Example:

```text
ChatGPT-like Behavior
```

---

# Architecture Changes

Pretraining:

```text
Standard GPT
```

---

Classification:

```text
Replace Output Layer
```

---

Instruction Tuning:

```text
Usually No Architecture Change
```

---

# Mental Model

Pretraining:

```text
Learn Language
```

Classification Fine-Tuning:

```text
Learn Labels
```

Instruction Fine-Tuning:

```text
Learn Behavior
```

---

# One Sentence Summary

Pretraining teaches a model how language works, classification fine-tuning teaches it how to assign labels, and instruction fine-tuning teaches it how to interact with humans and follow instructions.
