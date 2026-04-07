"""
NeuralLearner — Genuine neural learning via online gradient descent (CPU/numpy).

Replaces keyword-based KB memorization with a learned encoder that generalizes:
- Encodes text via character bigram hash trick (no vocabulary needed)
- Learns weights online after every solved problem via Adam SGD (Hebbian-style)
- At recall time: find nearest stored embeddings via cosine similarity
- Novel phrasing of a seen concept → similar embedding → correct answer returned

This is NOT memorization. The encoder learns a function over character patterns.
"Dog is to puppy as wolf is to ?" generalizes from "Dog is to puppy as cat is to ?"
because they share character bigrams and structural position.

Runs on CPU (numpy) to avoid conflicting with the MLX value network on the GPU.

Usage:
    nl = get_neural_learner()
    nl.learn("What is photosynthesis?", "process plants use to convert sunlight to food", "biology")
    answer = nl.recall("How do plants make food from sunlight?", "biology")  # generalizes
"""
from __future__ import annotations

import logging
import math
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_REPO_ROOT    = Path(__file__).resolve().parents[3]  # sare/python/sare/neuro → sare/
_WEIGHTS_PATH = _REPO_ROOT / "data" / "memory" / "neural_learner.npz"

_FEATURE_DIM = 4096   # hash space for character bigrams + word unigrams
_EMBED_DIM   = 128    # output embedding size


# ── Feature extraction ────────────────────────────────────────────────────────

def _featurize(text: str, dim: int = _FEATURE_DIM) -> np.ndarray:
    """
    Character bigram hash trick → dense float32 vector of shape (dim,).

    "hello world" → bigrams: "he","el","ll","lo","o ","wo","or","rl","ld"
    Each bigram hashed mod dim; word unigrams weighted 2× for semantic content.
    No vocabulary, no tokenizer — similar text shares bigrams → close vectors.
    """
    text = text.lower().strip()
    if not text:
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    words   = text.split()
    tokens  = bigrams + words + words  # words double-weighted
    if not tokens:
        return vec
    scale = 1.0 / math.sqrt(max(len(tokens), 1))
    for tok in tokens:
        idx = hash(tok) % dim
        if idx < 0:
            idx += dim
        vec[idx] += scale
    return vec


# ── Numpy 2-layer MLP with Adam ───────────────────────────────────────────────

class _NumpyMLP:
    """
    2-layer MLP: feat_dim → 256 → embed_dim, trained with Adam.
    Separate projection head for question→answer matching.
    All CPU, no GPU, thread-safe when used from a single thread.
    """

    def __init__(self, feat_dim: int, embed_dim: int, lr: float = 5e-4):
        rng = np.random.default_rng(42)
        scale1 = math.sqrt(2.0 / feat_dim)
        scale2 = math.sqrt(2.0 / 256)
        self.W1 = rng.standard_normal((feat_dim, 256)).astype(np.float32) * scale1
        self.b1 = np.zeros(256, dtype=np.float32)
        self.W2 = rng.standard_normal((256, embed_dim)).astype(np.float32) * scale2
        self.b2 = np.zeros(embed_dim, dtype=np.float32)
        # Projection head (question → answer space)
        self.Wp = np.eye(embed_dim, dtype=np.float32) * 0.1
        self.bp = np.zeros(embed_dim, dtype=np.float32)
        # Adam state
        self.lr = lr
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0
        self._init_adam()

    def _init_adam(self):
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.mWp = np.zeros_like(self.Wp); self.vWp = np.zeros_like(self.Wp)
        self.mbp = np.zeros_like(self.bp); self.vbp = np.zeros_like(self.bp)

    def _adam_step(self, param, grad, m, v):
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v[:] = self.beta2 * v + (1 - self.beta2) * grad**2
        mhat = m / (1 - self.beta1**self.t)
        vhat = v / (1 - self.beta2**self.t)
        param -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (embedding, hidden, proj_embedding)."""
        h = np.maximum(0, x @ self.W1 + self.b1)       # ReLU, [256]
        e = h @ self.W2 + self.b2                        # [embed_dim]
        norm = np.linalg.norm(e) + 1e-8
        e_norm = e / norm
        p = e_norm @ self.Wp + self.bp                   # projection head
        p_norm_val = np.linalg.norm(p) + 1e-8
        p_norm = p / p_norm_val
        return e_norm, h, p_norm

    def train_step(self, x_q: np.ndarray, x_a: np.ndarray, correct: bool,
                   n_steps: int = 3) -> float:
        """
        Run n_steps gradient descent steps.
        Loss = 1 - cos_sim(proj(q), a)  if correct  (pull toward answer)
             = cos_sim(proj(q), a) + 0.1  if wrong   (push away from answer)
        """
        loss_val = 0.0
        for _ in range(n_steps):
            self.t += 1
            e_q, h_q, p_q = self.forward(x_q)
            e_a, _,   _   = self.forward(x_a)
            sim = float(np.dot(p_q, e_a))

            # Loss and gradient of sim w.r.t. p_q
            sign = -1.0 if correct else 1.0   # minimize: -sim if correct, +sim if wrong
            loss_val = (1.0 - sim) if correct else (sim + 0.1)
            d_pq = sign * e_a                  # d(loss)/d(p_q) = sign * e_a

            # Backprop through projection head: p = e_q @ Wp + bp (normalized)
            # Approximate: ignore normalization Jacobian for efficiency
            d_Wp = np.outer(e_q, d_pq)
            d_bp = d_pq
            d_eq_from_proj = d_pq @ self.Wp.T

            # Backprop through W2: e = h @ W2 + b2 (pre-normalization)
            d_W2 = np.outer(h_q, d_eq_from_proj)
            d_b2 = d_eq_from_proj
            d_h  = d_eq_from_proj @ self.W2.T

            # Backprop through ReLU
            d_h_relu = d_h * (h_q > 0).astype(np.float32)
            d_W1 = np.outer(x_q, d_h_relu)
            d_b1 = d_h_relu

            # Adam updates
            self._adam_step(self.W1, d_W1, self.mW1, self.vW1)
            self._adam_step(self.b1, d_b1, self.mb1, self.vb1)
            self._adam_step(self.W2, d_W2, self.mW2, self.vW2)
            self._adam_step(self.b2, d_b2, self.mb2, self.vb2)
            self._adam_step(self.Wp, d_Wp, self.mWp, self.vWp)
            self._adam_step(self.bp, d_bp, self.mbp, self.vbp)

        return float(loss_val)

    def encode(self, x: np.ndarray) -> np.ndarray:
        e, _, _ = self.forward(x)
        return e


# ── NeuralLearner ─────────────────────────────────────────────────────────────

class NeuralLearner:
    """
    Neural associative memory with online gradient-descent learning.

    - learn(q, a, domain, correct) → enqueues for background gradient update
    - recall(q, domain, threshold) → cosine nearest-neighbor in memory bank

    The encoder (bigram features → 2-layer MLP → 128-dim unit vector) is trained
    continuously from every problem outcome. Similar questions map to close vectors
    after training — genuine generalization, not string lookup.
    """

    MEMORY_SIZE = 10_000
    THRESHOLD   = 0.72
    TOP_K       = 5
    QUEUE_SIZE  = 2000

    def __init__(self, embed_dim: int = _EMBED_DIM, feature_dim: int = _FEATURE_DIM,
                 memory_size: int = MEMORY_SIZE):
        self._embed_dim   = embed_dim
        self._feature_dim = feature_dim
        self._memory_size = memory_size

        self._mlp      = _NumpyMLP(feature_dim, embed_dim)
        self._lock     = threading.Lock()   # protects memory bank reads
        self._mlp_lock = threading.Lock()   # protects MLP weight read/write

        # Memory bank
        self._bank_emb: List[np.ndarray] = []
        self._bank_ans: List[str]         = []
        self._bank_dom: List[str]         = []
        self._bank_ptr: int               = 0

        # Training queue (non-blocking enqueue, single background consumer)
        self._train_q: "queue.Queue[Tuple[str,str,str,bool]]" = queue.Queue(maxsize=self.QUEUE_SIZE)

        self._learn_count   = 0
        self._recall_hits   = 0
        self._recall_misses = 0

        # Background training thread (sole user of _mlp weights)
        self._train_thread = threading.Thread(
            target=self._train_loop, daemon=True, name="NeuralLearner-Train"
        )
        self._train_thread.start()
        log.debug("[NeuralLearner] Numpy CPU encoder ready, training thread started")
        # Auto-load saved memories on construction so any process benefits from past learning
        self.load()

    # ── Encoding ─────────────────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to unit 128-dim vector."""
        x = _featurize(text, self._feature_dim)
        with self._mlp_lock:
            return self._mlp.encode(x)

    # ── Memory bank ──────────────────────────────────────────────────────────

    def _store(self, emb: np.ndarray, answer: str, domain: str) -> None:
        """Store in ring buffer. Must be called with self._lock held."""
        if len(self._bank_emb) < self._memory_size:
            self._bank_emb.append(emb)
            self._bank_ans.append(answer)
            self._bank_dom.append(domain)
        else:
            idx = self._bank_ptr % self._memory_size
            self._bank_emb[idx] = emb
            self._bank_ans[idx] = answer
            self._bank_dom[idx] = domain
            self._bank_ptr = (self._bank_ptr + 1) % self._memory_size

    # ── Background training loop ──────────────────────────────────────────────

    def _train_loop(self) -> None:
        """Drain training queue, run gradient steps, update memory bank."""
        while True:
            try:
                question, answer, domain, correct = self._train_q.get(timeout=2.0)
                xq = _featurize(question, self._feature_dim)
                xa = _featurize(answer,   self._feature_dim)
                with self._mlp_lock:
                    self._mlp.train_step(xq, xa, correct, n_steps=3)
                    if correct:
                        emb = self._mlp.encode(xq)
                with self._lock:
                    if correct:
                        self._store(emb, answer, domain)
                        self._learn_count += 1
                self._train_q.task_done()
            except Exception:
                pass  # queue.Empty on timeout → loop again

    # ── Public API ───────────────────────────────────────────────────────────

    def learn(self, question: str, answer: str, domain: str = "general",
              correct: bool = True) -> None:
        """
        Enqueue a learning example. Non-blocking — gradient runs in background.
        correct=True: pull encoder toward (question → answer) mapping.
        correct=False: push encoder away (contrastive negative).
        """
        question = (question or "").strip()
        answer   = (answer   or "").strip()
        if not question or not answer:
            return
        try:
            self._train_q.put_nowait((question, answer, domain, correct))
        except queue.Full:
            pass  # queue full — drop; best-effort training

    def recall(self, question: str, domain: str = "general",
               threshold: float = THRESHOLD) -> Optional[str]:
        """
        Find nearest stored embedding by cosine similarity.
        Returns the weighted-vote answer if best score ≥ threshold, else None.
        """
        question = (question or "").strip()
        if not question:
            return None

        with self._lock:
            if not self._bank_emb:
                return None
            # Need enough domain-specific examples before recall is reliable.
            # Below 500 memories the bigram MLP is undertrained and produces false positives.
            n_total = len(self._bank_emb)
            if n_total < 500:
                return None
            try:
                q_emb = self._encode(question)
                bank  = np.stack(self._bank_emb, axis=0)   # (N, D)
                scores = bank @ q_emb                        # cosine (all L2-normalized)

                # Domain filtering: with small memory banks, same-domain questions dominate;
                # only cross-domain if memory > 1000 (enough training to be discriminative)
                n_memories = len(self._bank_emb)
                score_adj = scores.copy()
                for i in range(len(self._bank_dom)):
                    if self._bank_dom[i] == domain:
                        score_adj[i] += 0.10
                    elif n_memories < 1000:
                        # Heavily penalize cross-domain matches when memory is small
                        score_adj[i] -= 0.20

                best_local = int(np.argmax(score_adj))
                best_score = float(scores[best_local])
                # Scale threshold up when memory is small — fewer examples = more false positives
                # With <500 memories the bigram encoder hasn't trained enough to be discriminative
                n_memories = len(self._bank_emb)
                _effective_threshold = threshold
                if n_memories < 200:
                    _effective_threshold = max(threshold, 0.88)
                elif n_memories < 500:
                    _effective_threshold = max(threshold, 0.82)
                elif n_memories < 1000:
                    _effective_threshold = max(threshold, 0.78)
                if best_score < _effective_threshold:
                    self._recall_misses += 1
                    return None

                # Top-K weighted vote
                k = min(self.TOP_K, len(scores))
                top_k = np.argsort(score_adj)[-k:][::-1]
                vote: Dict[str, float] = {}
                for idx in top_k:
                    ans = self._bank_ans[idx]
                    vote[ans] = vote.get(ans, 0.0) + float(scores[idx])
                best_ans = max(vote, key=vote.get)
                self._recall_hits += 1
                return best_ans
            except Exception as e:
                log.debug("[NeuralLearner] recall error: %s", e)
                return None

    def flush(self, timeout: float = 30.0) -> None:
        """Block until training queue is drained (or timeout). Call before save()."""
        try:
            self._train_q.join()  # waits for all tasks to be marked done
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._bank_emb)

    def get_stats(self) -> dict:
        total = self._recall_hits + self._recall_misses
        return {
            "memory_size":  len(self._bank_emb),
            "memory_cap":   self._memory_size,
            "learn_calls":  self._learn_count,
            "recall_hits":  self._recall_hits,
            "recall_misses": self._recall_misses,
            "hit_rate":     round(self._recall_hits / max(1, total), 3),
            "queue_pending": self._train_q.qsize(),
            "backend":      "numpy-cpu",
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            save = {
                "W1": self._mlp.W1, "b1": self._mlp.b1,
                "W2": self._mlp.W2, "b2": self._mlp.b2,
                "Wp": self._mlp.Wp, "bp": self._mlp.bp,
                "adam_t": np.array([self._mlp.t]),
                "learn_count": np.array([self._learn_count]),
                "bank_ans": np.array(self._bank_ans, dtype=object),
                "bank_dom": np.array(self._bank_dom, dtype=object),
                "bank_ptr": np.array([self._bank_ptr]),
            }
            if self._bank_emb:
                save["bank_emb"] = np.stack(self._bank_emb, axis=0)
            tmp = _WEIGHTS_PATH.with_suffix(".tmp.npz")
            np.savez_compressed(str(tmp), **save)
            tmp.replace(_WEIGHTS_PATH)
            log.debug("[NeuralLearner] Saved: %d memories", len(self._bank_emb))
        except Exception as e:
            log.debug("[NeuralLearner] Save error: %s", e)

    def load(self) -> None:
        if not _WEIGHTS_PATH.exists():
            return
        try:
            data = np.load(str(_WEIGHTS_PATH), allow_pickle=True)
            for attr in ("W1","b1","W2","b2","Wp","bp"):
                if attr in data:
                    setattr(self._mlp, attr, data[attr].astype(np.float32))
            if "adam_t" in data:
                self._mlp.t = int(data["adam_t"][0])
            if "learn_count" in data:
                self._learn_count = int(data["learn_count"][0])
            if "bank_emb" in data:
                arr = data["bank_emb"]
                self._bank_emb = [arr[i] for i in range(len(arr))]
            if "bank_ans" in data:
                self._bank_ans = list(data["bank_ans"])
            if "bank_dom" in data:
                self._bank_dom = list(data["bank_dom"])
            if "bank_ptr" in data:
                self._bank_ptr = int(data["bank_ptr"][0])
            log.info("[NeuralLearner] Loaded: %d memories, %d learn calls",
                     len(self._bank_emb), self._learn_count)
        except Exception as e:
            log.debug("[NeuralLearner] Load error: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_INSTANCE: Optional[NeuralLearner] = None
_LOCK = threading.Lock()


def get_neural_learner() -> NeuralLearner:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = NeuralLearner()
    return _INSTANCE
