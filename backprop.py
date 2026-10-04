import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Net:
    def __init__(self, sizes, seed=0, init_scale=0.5):
        rng = np.random.default_rng(seed)
        self.W = [rng.uniform(-init_scale, init_scale, (o, i + 1))
                  for i, o in zip(sizes[:-1], sizes[1:])]
        self.dW_prev = [np.zeros_like(w) for w in self.W]

    def forward(self, X):
        ys = [X]
        for W in self.W:
            y_in = np.hstack([ys[-1], np.ones((len(X), 1))])  # append always-on unit
            ys.append(sigmoid(y_in @ W.T))                    # eqs. 1 & 2
        return ys

    def error(self, X, D):
        return 0.5 * np.sum((self.forward(X)[-1] - D) ** 2)  # eq. 3

    def backward(self, ys, D):
        grads = [None] * len(self.W)
        dE_dy = ys[-1] - D                                    # eq. 4
        for l in reversed(range(len(self.W))):
            y = ys[l + 1]
            dE_dx = dE_dy * y * (1 - y)                       # eq. 5
            y_in = np.hstack([ys[l], np.ones((len(D), 1))])
            grads[l] = dE_dx.T @ y_in                         # eq. 6
            dE_dy = dE_dx @ self.W[l][:, :-1]                 # eq. 7 (skip bias column)
        return grads

    def train(self, X, D, eps=0.1, alpha=0.9, max_sweeps=20000, tol=1e-3, verbose=True):
        for t in range(max_sweeps):
            ys = self.forward(X)
            E = 0.5 * np.sum((ys[-1] - D) ** 2)
            if E < tol:
                break
            grads = self.backward(ys, D)
            for l in range(len(self.W)):
                self.dW_prev[l] = -eps * grads[l] + alpha * self.dW_prev[l]  # eq. 9
                self.W[l] += self.dW_prev[l]
            if verbose and t % 1000 == 0:
                print(f"  sweep {t:5d}  E = {E:.4f}")
        return t, E

def xor_demo():
    print("=== XOR (2-2-1) ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    D = np.array([[0], [1], [1], [0]], float)
    for seed in range(10):  # gradient descent can hit local minima; retry seeds
        net = Net([2, 2, 1], seed=seed, init_scale=1.0)
        sweeps, E = net.train(X, D, eps=0.5, alpha=0.9, max_sweeps=10000, verbose=False)
        if E < 1e-2:
            break
    print(f"  seed {seed}: converged in {sweeps} sweeps, E = {E:.5f}")
    print("  outputs:", net.forward(X)[-1].ravel().round(3))
    print("  hidden activations:\n", net.forward(X)[1].round(3))

def symmetry_demo():
    print("\n=== Symmetry detection (6-2-1) ===")
    X = np.array([[int(b) for b in format(n, "06b")] for n in range(64)], float)
    D = np.array([[1.0 if (x == x[::-1]).all() else 0.0] for x in X])
    for seed in range(30):
        net = Net([6, 2, 1], seed=seed, init_scale=0.3)
        sweeps, E = net.train(X, D, eps=0.1, alpha=0.9, max_sweeps=5000, tol=0.1, verbose=False)
        if E < 0.1:
            break
    print(f"  seed {seed}: converged in {sweeps} sweeps, E = {E:.4f}")
    pred = net.forward(X)[-1]
    print(f"  accuracy: {((pred > 0.5) == (D > 0.5)).mean() * 100:.1f}%")
    print("  hidden-unit input weights (6 inputs, then bias):")
    print(net.W[0].round(2))


if __name__ == "__main__":
    xor_demo()
    symmetry_demo()