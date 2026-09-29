"""Double-DQN agent implemented in pure NumPy (no PyTorch / TF needed).

Small MLP (obs -> 256 -> 256 -> n_actions), Huber loss, Adam optimizer,
target network with Polyak averaging, epsilon-greedy exploration.
Everything checkpoints to disk so training is fully resumable.
"""
import os
import pickle
import tempfile

import numpy as np

from . import config as C


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, rng):
        self.capacity = capacity
        self.rng = rng
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def __len__(self):
        return self.size

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = self.rng.integers(0, self.size, size=n)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx])

    def save(self, path):
        n = self.size
        tmp = path + ".tmp"
        np.savez_compressed(
            tmp, s=self.s[:n], a=self.a[:n], r=self.r[:n],
            s2=self.s2[:n], d=self.d[:n])
        os.replace(tmp + ".npz", path)

    def load(self, path):
        if not os.path.exists(path):
            return False
        try:
            z = np.load(path)
            n = int(z["s"].shape[0])
            n = min(n, self.capacity)
            self.s[:n] = z["s"][:n]
            self.a[:n] = z["a"][:n]
            self.r[:n] = z["r"][:n]
            self.s2[:n] = z["s2"][:n]
            self.d[:n] = z["d"][:n]
            self.size = n
            self.ptr = n % self.capacity
            return True
        except Exception:
            return False


class MLP:
    """Simple fully connected net with ReLU hidden layers."""

    def __init__(self, sizes, rng):
        self.sizes = list(sizes)
        self.L = len(self.sizes) - 1
        self.W = []
        self.b = []
        for nin, nout in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append((rng.standard_normal((nin, nout)) * np.sqrt(2.0 / nin)))
            self.b.append(np.zeros(nout))

    def forward(self, x):
        """returns (output, cache) where cache[i] = (a_in, z_i) per layer."""
        cache = []
        a = x
        for i in range(self.L):
            z = a @ self.W[i] + self.b[i]
            cache.append((a, z))
            if i < self.L - 1:
                a = np.maximum(z, 0.0)
            else:
                a = z
        return a, cache

    def backward(self, cache, d_out):
        """Gradients given dL/d(output). Returns (grads_W, grads_b)."""
        gW = [None] * self.L
        gb = [None] * self.L
        delta = d_out
        for i in reversed(range(self.L)):
            a_in, z = cache[i]
            gW[i] = a_in.T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                da = delta @ self.W[i].T
                _, z_prev = cache[i - 1]
                delta = da * (z_prev > 0)
        return gW, gb

    def copy_from(self, other, tau=1.0):
        for i in range(self.L):
            self.W[i] = (1.0 - tau) * self.W[i] + tau * other.W[i]
            self.b[i] = (1.0 - tau) * self.b[i] + tau * other.b[i]


class Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        """Adam update with global-norm gradient clipping."""
        self.t += 1
        b1t = 1.0 - self.b1 ** self.t
        b2t = 1.0 - self.b2 ** self.t
        total_sq = 0.0
        for g in grads:
            total_sq += float(np.sum(g * g))
        total_norm = float(np.sqrt(total_sq))
        scale = 1.0
        if total_norm > C.GRAD_CLIP > 0:
            scale = C.GRAD_CLIP / (total_norm + 1e-12)
        for i, (p, g) in enumerate(zip(params, grads)):
            if scale != 1.0:
                g = g * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / b1t
            vhat = self.v[i] / b2t
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def state(self):
        return {"m": self.m, "v": self.v, "t": self.t}

    def load_state(self, st):
        self.m = [np.asarray(x) for x in st["m"]]
        self.v = [np.asarray(x) for x in st["v"]]
        self.t = int(st["t"])


class DQNAgent:
    def __init__(self, obs_dim, n_actions, rng, hidden=C.HIDDEN_LAYERS,
                 lr=C.LEARNING_RATE):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rng = rng
        sizes = [obs_dim] + list(hidden) + [n_actions]
        self.online = MLP(sizes, rng)
        self.target = MLP(sizes, rng)
        self.target.copy_from(self.online, tau=1.0)
        self.opt = Adam(self.params(), lr)
        self.total_steps = 0
        self.eps = C.EPS_START

    def params(self):
        return self.online.W + self.online.b

    # ---------------------------------------------------------------- acting
    def epsilon(self):
        frac = min(1.0, self.total_steps / max(1, C.EPS_DECAY_STEPS))
        return max(C.EPS_MIN, C.EPS_START + frac * (C.EPS_MIN - C.EPS_START))

    def act(self, obs, eps=None):
        if eps is None:
            eps = self.epsilon()
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        q, _ = self.online.forward(obs[None, :])
        return int(np.argmax(q[0]))

    # ---------------------------------------------------------------- learning
    def train_batch(self, s, a, r, s2, d):
        B = s.shape[0]
        q, cache = self.online.forward(s)
        qsa = q[np.arange(B), a]

        with np.errstate(all="ignore"):
            q2_online, _ = self.online.forward(s2)
            a_star = np.argmax(q2_online, axis=1)
            q2_target, _ = self.target.forward(s2)
            y = r + (1.0 - d) * C.GAMMA * q2_target[np.arange(B), a_star]

        diff = qsa - y
        # Huber (delta=1) gradient wrt selected q values, averaged over batch
        dq_sel = np.clip(diff, -1.0, 1.0) / B
        d_out = np.zeros_like(q)
        d_out[np.arange(B), a] = dq_sel

        gW, gb = self.online.backward(cache, d_out)
        grads = gW + gb
        self.opt.step(self.params(), grads)
        self.target.copy_from(self.online, tau=C.TARGET_TAU)
        self.total_steps += 1
        loss = float(np.mean(np.minimum(np.abs(diff), 0.5 * diff * diff)))
        td = float(np.mean(np.abs(diff)))
        return loss, td

    # ---------------------------------------------------------------- io
    def save(self, path, extra=None):
        state = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "W": self.online.W, "b": self.online.b,
            "Wt": self.target.W, "bt": self.target.b,
            "adam": self.opt.state(),
            "total_steps": self.total_steps,
            "eps": self.eps,
            "rng": self.rng.bit_generator.state,
            "extra": extra or {},
        }
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=4)
        os.replace(tmp, path)

    def load(self, path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state["obs_dim"] != self.obs_dim:
            return None
        self.online.W = [np.asarray(w) for w in state["W"]]
        self.online.b = [np.asarray(b) for b in state["b"]]
        self.target.W = [np.asarray(w) for w in state["Wt"]]
        self.target.b = [np.asarray(b) for b in state["bt"]]
        self.opt.load_state(state["adam"])
        self.total_steps = int(state["total_steps"])
        self.eps = float(state.get("eps", C.EPS_START))
        try:
            self.rng.bit_generator.state = state["rng"]
        except Exception:
            pass
        return state.get("extra") or {}
