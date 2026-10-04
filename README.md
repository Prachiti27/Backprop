# Backpropagation from Scratch

A simple NumPy implementation of the backpropagation algorithm from:

**Rumelhart, Hinton & Williams (1986)** — *Learning representations by back-propagating errors*, Nature 323, 533–536.

The implementation follows the equations and notation from the original paper.

## What it implements

- Feed-forward neural network
- Sigmoid activation
- Bias weights
- Backpropagation using the chain rule
- Batch gradient descent
- Momentum-based weight updates

## Experiments

### 1. XOR

Uses a `2 → 2 → 1` network to learn XOR.

A hidden layer is required because XOR cannot be solved by a single linear layer.

### 2. Symmetry Detection

Uses a `6 → 2 → 1` network to determine whether a 6-bit input is symmetric about its center.

This reproduces the type of experiment discussed in the original paper and allows inspection of the learned hidden-layer weights.

## Run

```bash
pip install numpy
python backprop.py
```

No deep-learning framework is used — the algorithm is implemented directly with NumPy.