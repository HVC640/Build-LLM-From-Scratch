# Embedding Training Intuition

## The Mystery

One of the most confusing parts of LLMs is:

```text
How does the model learn embeddings?
```

Nobody tells GPT:

```text
dog = animal

cat = animal

car = vehicle
```

There is no dictionary of meanings.

There is no teacher assigning semantic relationships.

Yet after training:

```text
dog
cat
puppy
```

end up close together in embedding space.

Why?

---

# The Common Misconception

Many beginners imagine:

```text
Tokenizer
 ↓

Train Embeddings
 ↓

Train GPT
```

as separate stages.

This is usually incorrect.

GPT trains:

```text
Embeddings
Attention
FFN
Output Layer
```

all at the same time.

Embeddings are simply another set of trainable parameters.

---

# What Does An Embedding Layer Really Contain?

Suppose:

```text
Vocabulary Size = 50,000

Embedding Dimension = 768
```

The embedding layer is simply:

```text
50,000 × 768
```

trainable numbers.

Example:

```text
dog
```

might initially map to:

```text
[0.12, -0.88, 0.41, ...]
```

which is completely random.

At initialization:

```text
dog
cat
car
banana
```

have no meaningful relationship.

---

# The Key Insight

Embeddings are NOT trained to understand words.

Embeddings are trained to reduce prediction error.

This is extremely important.

The model does not ask:

```text
What does "dog" mean?
```

The model asks:

```text
How should this vector change
to improve next-token prediction?
```

Meaning emerges as a side effect.

---

# Example

Suppose the training data contains:

```text
I love dogs

Dogs are friendly

My dog is happy
```

and

```text
I love cats

Cats are friendly

My cat is happy
```

Notice something:

```text
dog
```

and

```text
cat
```

appear in very similar contexts.

---

# What Happens During Training?

Initially:

```text
dog → random vector

cat → random vector
```

The model makes poor predictions.

Cross Entropy Loss becomes large.

Backpropagation computes gradients.

---

The optimizer then updates:

```text
dog vector

cat vector
```

so future predictions improve.

This process repeats millions of times.

---

# Why Do Similar Words Become Similar?

Because they receive similar gradient updates.

Consider:

```text
My dog is happy

My cat is happy
```

Both sentences create similar learning signals.

Over time:

```text
dog
```

and

```text
cat
```

are pushed toward similar regions.

Not because anyone explicitly grouped them.

But because similar contexts create similar gradients.

---

# Distributional Hypothesis

A famous NLP principle states:

> Words that occur in similar contexts tend to have similar meanings.

Example:

```text
The dog barked

The cat meowed
```

and

```text
My dog is cute

My cat is cute
```

The model repeatedly observes:

```text
dog
```

and

```text
cat
```

behaving similarly.

Therefore their embeddings become similar.

---

# Why Does Meaning Emerge?

The model discovers:

```text
Words that help make similar predictions
should have similar representations.
```

This creates semantic structure naturally.

Nobody explicitly teaches:

```text
dog = animal
```

The model infers it from usage patterns.

---

# What Does Backpropagation Update?

Suppose:

```text
Input:
I love dogs
```

Target:

```text
very much
```

Prediction is poor.

Loss becomes high.

Gradients flow:

```text
Loss
 ↓

Output Layer
 ↓

Transformer Blocks
 ↓

Embedding Layer
```

Eventually:

```text
Embedding("dogs")
```

receives an update.

This update slightly changes the vector.

After billions of updates:

```text
Meaningful embeddings emerge.
```

---

# Relationship With Attention

Embeddings provide the starting representation.

Attention then enriches that representation.

Example:

Initial embedding:

```text
bank
```

could mean:

```text
river bank

financial bank
```

The embedding alone is ambiguous.

---

After attention:

```text
bank
```

inside:

```text
I deposited money in the bank
```

becomes a contextualized representation.

This richer representation is what GPT actually uses.

---

# Static vs Contextual Meaning

Embedding Layer:

```text
One vector per token.
```

Example:

```text
bank
```

always starts with the same vector.

---

After Transformer Layers:

```text
bank
```

becomes context-dependent.

Example:

```text
river bank
```

and

```text
financial bank
```

produce different final representations.

---

# Mental Model

Think of embeddings as:

```text
Raw Ingredients
```

Attention as:

```text
Cooking Process
```

Transformer Blocks as:

```text
The Entire Kitchen
```

Embeddings provide the starting material.

Training gradually improves those ingredients.

---

# Common Interview Question

Q:

```text
Are embeddings trained separately in GPT?
```

A:

```text
No.

Embeddings are trained jointly with the entire model.

They receive gradients through backpropagation just like attention weights, FFN weights, and output layer weights.
```

---

# One Sentence Summary

Embeddings start as random vectors and become meaningful because backpropagation repeatedly adjusts them to improve next-token prediction; semantic relationships emerge naturally because words appearing in similar contexts receive similar gradient updates.
