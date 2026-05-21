---
title: "Neural Networks from First Principles"
date: 2026-05-21 10:00:00 +0700
categories: [ml]
tags: [neural-networks, math, backprop]
---

A condensed walkthrough of the core math behind a feed-forward neural network, written for someone learning for the first time. Companion notes to the from-scratch Go implementation in this repo.

The whole network reduces to **four ideas**:

1. A neuron is a weighted sum + bias + non-linearity.
2. A layer is the same thing in matrix form.
3. A loss function turns a prediction into a single "how wrong" number.
4. Backpropagation computes how to nudge every parameter to make that number smaller.

That's it. Everything else is engineering.

---

## Step 1 — A single neuron

A neuron takes inputs `x₁, …, xₙ`, multiplies each by a learned **weight** `wᵢ`, sums them, adds a **bias** `b`, then passes the result through a non-linear **activation function** `σ`:

```
z = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b
a = σ(z)
```

- `z` is the **pre-activation**.
- `a` is the **activation** — the neuron's output.

### Why the activation function?

Without `σ`, stacking layers would just be one big linear function — no matter how deep, the network could only learn straight-line relationships. The non-linearity is what lets the network learn curves, edges, digits, anything interesting.

Common choices:

- **Sigmoid**: `σ(z) = 1 / (1 + e⁻ᶻ)` — smooth, squashes to `(0, 1)`. Classic, slow.
- **ReLU**: `σ(z) = max(0, z)` — fast, works great in practice. Default for hidden layers.

### Tiny example

One neuron, 2 inputs:
- `x = [1.0, 2.0]`
- `w = [0.5, -0.3]`
- `b = 0.1`

```
z = 0.5·1.0 + (-0.3)·2.0 + 0.1 = 0.0
a = ReLU(0.0) = 0.0
```

A **layer** is just many neurons in parallel, each with its own weights and bias, all reading the same input.

---

## Step 2 — Forward pass for a layer

Writing neuron-by-neuron gets messy. Pack everything into matrices.

### Setup

3 neurons, 2 inputs. Each neuron has 2 weights and 1 bias.

Neuron-by-neuron:
```
z₁ = w₁₁·x₁ + w₁₂·x₂ + b₁
z₂ = w₂₁·x₁ + w₂₂·x₂ + b₂
z₃ = w₃₁·x₁ + w₃₂·x₂ + b₃
```

Notation: `wᵢⱼ` = weight from input `j` into neuron `i`.

### Same thing as matrix math

Stack the weights into a matrix `W` (one row per neuron):

```
W = [ w₁₁  w₁₂ ]     b = [ b₁ ]     x = [ x₁ ]
    [ w₂₁  w₂₂ ]         [ b₂ ]         [ x₂ ]
    [ w₃₁  w₃₂ ]         [ b₃ ]
```

The whole layer's computation is:

```
z = W·x + b           (shape: 3×1)
a = σ(z)              (element-wise)
```

**One matrix multiply + one vector add + one element-wise function = a full layer.**

### Numerical example

```
W = [ 0.5  -0.3 ]    b = [ 0.1 ]    x = [ 1.0 ]
    [ 0.2   0.8 ]        [ 0.0 ]        [ 2.0 ]
    [-0.1   0.4 ]        [ 0.2 ]
```

```
z₁ = 0.5·1.0 + (-0.3)·2.0 + 0.1 =  0.0
z₂ = 0.2·1.0 +   0.8·2.0  + 0.0 =  1.8
z₃ = -0.1·1.0 +  0.4·2.0  + 0.2 =  0.9

a = ReLU(z) = [0.0, 1.8, 0.9]
```

### Chaining layers

A network is just this, repeated:

```
a⁽¹⁾ = σ(W⁽¹⁾·x    + b⁽¹⁾)
a⁽²⁾ = σ(W⁽²⁾·a⁽¹⁾ + b⁽²⁾)
a⁽³⁾ = σ(W⁽³⁾·a⁽²⁾ + b⁽³⁾)   ← final output
```

### Shape cheat-sheet

For layer `ℓ` with `nₗ` neurons reading `nₗ₋₁` inputs:

| object         | shape          |
|----------------|----------------|
| `W⁽ˡ⁾`         | `nₗ × nₗ₋₁`    |
| `b⁽ˡ⁾`         | `nₗ × 1`       |
| input to layer | `nₗ₋₁ × 1`     |
| output         | `nₗ × 1`       |

For MNIST with one hidden layer of 128:
- `W⁽¹⁾: 128 × 784`, `b⁽¹⁾: 128 × 1`
- `W⁽²⁾: 10 × 128`, `b⁽²⁾: 10 × 1`

---

## Step 3 — Loss function

The forward pass gives us a prediction. The **loss** is a single number that says how wrong it was. Training = make this number small.

### Part A — Softmax (raw scores → probabilities)

The final layer outputs 10 arbitrary numbers (positive, negative, large). **Softmax** turns them into a probability distribution:

```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

It exponentiates (making everything positive) and normalizes.

Example with 3 classes, `z = [2.0, 1.0, 0.1]`:

```
exp(2.0) = 7.389
exp(1.0) = 2.718
exp(0.1) = 1.105
sum      = 11.212

p = [7.389/11.212, 2.718/11.212, 1.105/11.212]
  = [0.659, 0.242, 0.099]
```

They sum to 1.0 — a real probability distribution.

> **Implementation note**: subtract `max(z)` before `exp` to avoid overflow. Same result mathematically (the constant cancels), but `exp(1000)` is `+Inf` otherwise.

### Part B — One-hot encoding the label

The true class index becomes a vector with a 1 at the right slot, 0s elsewhere:

```
true class = 0  →  y = [1, 0, 0]
true class = 2  →  y = [0, 0, 1]
```

Same shape as the prediction so we can compare directly.

### Part C — Cross-entropy loss

```
L = -Σᵢ yᵢ · log(pᵢ)
```

Because `y` is one-hot, this collapses to:

```
L = -log(p_correct)
```

where `p_correct` is the probability the network assigned to the true class.

Example: true class is 0, `p_correct = 0.659`:

```
L = -log(0.659) = 0.417
```

- Confident-right (`p = 0.95`): `L = 0.051` ← small, good
- Confident-wrong (`p = 0.05`): `L = 2.996` ← huge, very bad

### Why this loss?

1. **It punishes confident-wrong answers extremely hard.** `-log(p)` shoots to infinity as `p → 0`. A network that's 99% sure of the wrong answer pays a huge price.
2. **The gradient is incredibly clean.** Combine softmax + cross-entropy and the gradient w.r.t. the pre-softmax scores `z` is just:
   ```
   ∂L/∂z = p - y
   ```
   *Predicted minus true.* This is why softmax and cross-entropy are always paired.

---

## Step 4 — Backpropagation

Goal: for every weight `W` and bias `b`, compute `∂L/∂W` and `∂L/∂b` — how the loss changes if we nudge that parameter — then step against the gradient:

```
W ← W - η·∂L/∂W
b ← b - η·∂L/∂b
```

`η` (eta) is the **learning rate**. Small number like `0.01`.

### The chain rule, applied backwards

Loss depends on the output, which depends on the previous layer, which depends on the one before — chain rule lets us compute gradients piece by piece, walking *backward*.

The efficient algorithm for this is **backpropagation**.

### What to save during the forward pass

For each layer `ℓ`:
- `a⁽ˡ⁻¹⁾` — the input to that layer
- `z⁽ˡ⁾`   — the pre-activation
- `a⁽ˡ⁾`   — the activation

These are the ingredients for the backward pass.

### The recipe (3 lines, repeated per layer)

For each layer `ℓ`, going from output toward input:

```
δ⁽ˡ⁾    = (W⁽ˡ⁺¹⁾)ᵀ · δ⁽ˡ⁺¹⁾  ⊙  σ'(z⁽ˡ⁾)        ← chain through the activation
∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

Symbols:
- `δ⁽ˡ⁾` = `∂L/∂z⁽ˡ⁾`, the "error" at layer `ℓ`.
- `⊙` = element-wise (Hadamard) product.
- `σ'(z)` = derivative of the activation, e.g. `ReLU'(z) = 1 if z > 0 else 0`.

### The boundary condition (where backprop starts)

For the output layer, thanks to the softmax + cross-entropy identity from Step 3:

```
δ⁽ᴸ⁾ = p − y
```

That's the whole starting point. *Predicted minus true.* If the output layer uses a Linear (identity) activation (which is what we do, since softmax is applied by the loss), the formula above is consistent: `δ = ∂L/∂a ⊙ σ'(z) = (p − y) ⊙ 1`.

### Full backprop for a 2-layer network

Architecture: `x → [W⁽¹⁾, b⁽¹⁾, ReLU] → a⁽¹⁾ → [W⁽²⁾, b⁽²⁾, Linear] → z⁽²⁾ → softmax → p → L`

```
δ⁽²⁾     = p - y
∂L/∂W⁽²⁾ = δ⁽²⁾ · (a⁽¹⁾)ᵀ
∂L/∂b⁽²⁾ = δ⁽²⁾

δ⁽¹⁾     = (W⁽²⁾)ᵀ · δ⁽²⁾  ⊙  ReLU'(z⁽¹⁾)
∂L/∂W⁽¹⁾ = δ⁽¹⁾ · xᵀ
∂L/∂b⁽¹⁾ = δ⁽¹⁾
```

Then update everything:

```
W⁽ˡ⁾ ← W⁽ˡ⁾ - η·∂L/∂W⁽ˡ⁾
b⁽ˡ⁾ ← b⁽ˡ⁾ - η·∂L/∂b⁽ˡ⁾
```

**Important**: compute all gradients first, *then* apply the updates. If you mutate `W⁽²⁾` before computing `δ⁽¹⁾`, you'll multiply by the updated weights — wrong.

### Why is it called "back" propagation?

You *could* derive each gradient from scratch using the full chain rule. But you'd recompute the same intermediate products over and over. By walking backward and reusing `δ⁽ˡ⁺¹⁾` to compute `δ⁽ˡ⁾`, the total cost is about the same as a single forward pass.

That efficiency is what made deep learning practical.

---

## The whole training loop in one paragraph

For each training example `(x, y)`:
1. **Forward pass**: walk layers, compute `a⁽ˡ⁾`, cache `xIn` and `z⁽ˡ⁾`.
2. **Loss**: `p = softmax(z_last)`, `L = -log(p_correct)`.
3. **Backward pass**: start with `δ = p − y`. Walk layers in reverse:
   - record `dW⁽ˡ⁾ = δ · xInᵀ`, `db⁽ˡ⁾ = δ`
   - propagate: `δ ← (W⁽ˡ⁾)ᵀ · δ ⊙ σ'(z⁽ˡ⁻¹⁾)`
4. **Update**: `W -= η·dW`, `b -= η·db` for every layer.

Repeat thousands of times. The network learns.

---

## Mental picture

- **Forward pass**: data flows in → predictions flow out.
- **Loss**: measures the error.
- **Backward pass**: error flows from output back through the network — at each layer we ask "how much did *you* contribute to this error?" That's the gradient.
- **Update**: every weight steps a tiny bit in the direction that reduces error.

Everything else in modern deep learning (convolutions, attention, batch norm, optimizers like Adam) is variation and refinement on top of these four ideas.

---

## Glossary

| term            | meaning                                                          |
|-----------------|------------------------------------------------------------------|
| weight `W`      | learned parameter; multiplies an input                           |
| bias `b`        | learned offset added after the weighted sum                      |
| pre-activation `z` | the linear output `W·x + b` before the non-linearity          |
| activation `a`  | the output of a neuron, `σ(z)`                                   |
| σ (sigma)       | activation function, e.g. ReLU or Sigmoid                        |
| σ'              | derivative of the activation                                     |
| forward pass    | computing predictions layer by layer                             |
| loss `L`        | scalar measure of how wrong the prediction is                    |
| softmax         | turns raw scores into a probability distribution                 |
| one-hot         | label encoded as a vector with a single 1                        |
| cross-entropy   | classification loss: `-log(p_correct)`                           |
| backprop        | algorithm for computing gradients by walking the chain rule backward |
| δ (delta)       | `∂L/∂z`, the error signal at a layer                            |
| `∂L/∂W`         | gradient of loss w.r.t. weights — tells us how to nudge W       |
| learning rate η | step size for the gradient update                                |
| SGD             | stochastic gradient descent — update on one example at a time    |
| epoch           | one full pass through the training data                          |

---

## How this maps to the code in this repo

| concept                          | file                |
|----------------------------------|---------------------|
| matrix math (Dot, Add, Hadamard) | `matrix/matrix.go`  |
| activation + its derivative      | `nn/activations.go` |
| forward pass, backward, SGD step | `nn/layer.go`       |
| softmax + cross-entropy          | `nn/loss.go`        |
| MNIST loader (IDX format)        | `mnist/mnist.go`    |
| training loop                    | `cmd/train/main.go` |

Run `go test ./...` — the tests verify the exact numerical examples above (Step 2's `z = [0, 1.8, 0.9]`, Step 3's `p ≈ [0.659, 0.242, 0.099]`, etc.). The XOR test in `nn/train_test.go` is a tiny end-to-end proof that forward + backward + update are all correct.

Run `go run ./cmd/train` to actually train on MNIST. Expect ~97% test accuracy after 3 epochs.
