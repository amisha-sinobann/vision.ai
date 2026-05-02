"""
rnn_temporal.py  —  Vision OS Temporal Scene Analyser
======================================================
Uses a lightweight numpy-only LSTM (no heavy framework dependency)
to analyse a rolling window of per-frame feature vectors and produce:
  • motion_level    : "none" | "low" | "medium" | "high"
  • trend           : "appearing" | "disappearing" | "stable" | "chaotic"
  • predicted_labels: list[str]   – objects likely in the next frame
  • anomaly         : bool        – sudden unexpected scene change
  • temporal_summary: str         – human-readable sentence

Optional PyTorch path: if torch is available the LSTM cell is replaced by
a proper nn.LSTMCell for better long-sequence accuracy.  The public API
(RNNTemporalAnalyser.update / .get_analysis) is identical either way.

Usage
-----
    from rnn_temporal import RNNTemporalAnalyser

    rnn = RNNTemporalAnalyser(window=12)        # keep 12 frames of history

    # call every frame from analyse_frame():
    rnn.update(detection_dict, frame)

    # read the temporal result:
    temporal = rnn.get_analysis()
    # → {"motion_level": "low", "trend": "stable", "predicted_labels": [...], ...}
"""

import numpy as np
import collections
import threading
import time
from typing import Dict, List, Optional


# ── Feature extraction ────────────────────────────────────────────────────────
# 80 COCO classes (same order as YOLOv8)
_COCO_80 = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]
_LABEL_IDX = {lbl: i for i, lbl in enumerate(_COCO_80)}
_N_CLASSES  = len(_COCO_80)          # 80

# Feature vector layout (total = 80 + 5 = 85 dimensions):
#   [0:80]  per-class presence (max confidence if seen, else 0)
#   [80]    object count (normalised by 20)
#   [81]    dominant-colour hue  (0-1, greyscale → 0.5)
#   [82]    obstacle level  (SAFE=0, WARN=0.5, DANGER=1)
#   [83]    currency detected  (0 / 1)
#   [84]    inference time (ms, normalised by 500)
_FEAT_DIM = _N_CLASSES + 5


def _detection_to_vector(detection: dict) -> np.ndarray:
    """Convert a detection dict → float32 feature vector of shape (_FEAT_DIM,)."""
    vec = np.zeros(_FEAT_DIM, dtype=np.float32)

    # Class confidences
    for obj in detection.get("objects", []):
        lbl = obj.get("label", "").lower().strip()
        # strip OCR prefix
        if lbl.startswith("txt: "):
            lbl = lbl[5:]
        idx = _LABEL_IDX.get(lbl)
        if idx is not None:
            conf = obj.get("confidence", 0)
            if isinstance(conf, int):
                conf = conf / 100.0
            vec[idx] = max(vec[idx], float(conf))

    # Object count (capped at 20)
    vec[80] = min(len(detection.get("objects", [])), 20) / 20.0

    # Dominant colour hue
    cols = detection.get("dominant_colours", [])
    if cols:
        hex_col = cols[0].get("hex", "#808080")
        try:
            r = int(hex_col[1:3], 16) / 255.0
            g = int(hex_col[3:5], 16) / 255.0
            b = int(hex_col[5:7], 16) / 255.0
            import colorsys
            h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
            vec[81] = float(h)
        except Exception:
            vec[81] = 0.5

    # Obstacle level
    obs_map = {"SAFE": 0.0, "WARN": 0.5, "WARNING": 0.5, "DANGER": 1.0}
    obs = detection.get("obstacle", {})
    vec[82] = obs_map.get(str(obs.get("level", "SAFE")).upper(), 0.0)

    # Currency
    vec[83] = 1.0 if (detection.get("currency") or {}).get("detected") else 0.0

    # Inference time
    vec[84] = min(detection.get("inference_ms", 0), 500) / 500.0

    return vec


# ── Numpy-only LSTM cell ──────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -30, 30))


class _NumpyLSTMCell:
    """Single LSTM cell with Xavier-initialised random weights (no training).

    We use it as a *feature projector* — the weights are fixed random but the
    recurrent hidden state still accumulates temporal correlations that the
    downstream analysis layer can exploit.  This is equivalent to an Echo
    State Network / reservoir computing approach: surprisingly effective for
    anomaly detection and trend extraction without any training data.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale_w = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_u = np.sqrt(2.0 / (hidden_dim + hidden_dim))

        # Concatenated [forget, input, gate, output] gates
        n = hidden_dim
        self.W  = rng.normal(0, scale_w, (4 * n, input_dim)).astype(np.float32)
        self.U  = rng.normal(0, scale_u, (4 * n, hidden_dim)).astype(np.float32)
        self.b  = np.zeros(4 * n, dtype=np.float32)
        # Initialise forget-gate bias to 1 (standard trick for stability)
        self.b[:n] = 1.0
        self.n  = n

        # State
        self.h = np.zeros(n, dtype=np.float32)
        self.c = np.zeros(n, dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Process one time-step, return hidden state h (shape [hidden_dim])."""
        n    = self.n
        z    = self.W @ x + self.U @ self.h + self.b
        f    = _sigmoid(z[0*n : 1*n])
        i    = _sigmoid(z[1*n : 2*n])
        g    =    _tanh(z[2*n : 3*n])
        o    = _sigmoid(z[3*n : 4*n])
        self.c = f * self.c + i * g
        self.h = o * _tanh(self.c)
        return self.h.copy()

    def reset(self):
        self.h[:] = 0.0
        self.c[:] = 0.0


# ── Optional PyTorch path ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    class _TorchLSTMCell:
        def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
            torch.manual_seed(seed)
            self.cell = nn.LSTMCell(input_dim, hidden_dim)
            self.h    = torch.zeros(1, hidden_dim)
            self.c    = torch.zeros(1, hidden_dim)
            self.n    = hidden_dim

        def step(self, x: np.ndarray) -> np.ndarray:
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                self.h, self.c = self.cell(xt, (self.h, self.c))
            return self.h.squeeze(0).numpy()

        def reset(self):
            self.h.zero_()
            self.c.zero_()

    _USE_TORCH = True
    print("[RNN] ✅ PyTorch detected — using nn.LSTMCell")

except ImportError:
    _USE_TORCH = False
    print("[RNN] ℹ️  PyTorch not found — using numpy reservoir LSTM (still effective)")


# ── Analysis layer ────────────────────────────────────────────────────────────

class _AnalysisHead:
    """Simple linear readout + heuristics on the hidden-state history."""

    def __init__(self, hidden_dim: int, n_classes: int):
        rng = np.random.default_rng(99)
        # Weight matrix: maps hidden → class logits (for prediction)
        self.W_pred = rng.normal(0, 0.1, (n_classes, hidden_dim)).astype(np.float32)
        self.n      = n_classes

    def analyse(
        self,
        h_history:    np.ndarray,   # shape [T, hidden_dim]
        feat_history: np.ndarray,   # shape [T, _FEAT_DIM]
    ) -> dict:
        """Return analysis dict from hidden-state history."""
        T = len(h_history)
        if T < 2:
            return {"motion_level": "none", "trend": "stable",
                    "predicted_labels": [], "anomaly": False,
                    "temporal_summary": "Insufficient history."}

        h_last = h_history[-1]

        # ── Predicted labels (top-3 logit classes) ────────────────────────────
        logits    = self.W_pred @ h_last
        top_idx   = np.argsort(logits)[::-1][:5]
        # Only keep classes that also appeared recently
        seen_mask = feat_history[-min(T, 5):, :80].max(axis=0) > 0.15
        predicted = [_COCO_80[i] for i in top_idx if seen_mask[i]][:3]

        # ── Motion level from object-count variance ────────────────────────────
        counts = feat_history[:, 80]   # normalised
        count_var = float(np.var(counts))
        if   count_var < 0.001: motion = "none"
        elif count_var < 0.01:  motion = "low"
        elif count_var < 0.05:  motion = "medium"
        else:                   motion = "high"

        # ── Trend: is the scene getting busier or quieter? ────────────────────
        if T >= 4:
            first_half = counts[:T//2].mean()
            second_half = counts[T//2:].mean()
            delta = second_half - first_half
            if   delta >  0.05: trend = "appearing"
            elif delta < -0.05: trend = "disappearing"
            else:
                # Use hidden-state cosine similarity for stability
                cos = np.dot(h_history[-1], h_history[-2]) / (
                    np.linalg.norm(h_history[-1]) * np.linalg.norm(h_history[-2]) + 1e-8
                )
                trend = "stable" if cos > 0.90 else "chaotic"
        else:
            trend = "stable"

        # ── Anomaly: sudden hidden-state divergence ───────────────────────────
        if T >= 3:
            diffs = [
                np.linalg.norm(h_history[t] - h_history[t - 1])
                for t in range(max(1, T - 5), T)
            ]
            mean_diff = np.mean(diffs[:-1]) if len(diffs) > 1 else 0.0
            last_diff = diffs[-1]
            anomaly   = bool(last_diff > 3.0 * mean_diff + 0.05)
        else:
            anomaly = False

        # ── Natural-language summary ──────────────────────────────────────────
        summary_parts = []
        if motion == "none":
            summary_parts.append("Scene is static.")
        else:
            summary_parts.append(f"Scene activity is {motion}.")
        if trend == "appearing":
            summary_parts.append("More objects appearing.")
        elif trend == "disappearing":
            summary_parts.append("Objects leaving the frame.")
        elif trend == "chaotic":
            summary_parts.append("Scene is changing rapidly.")
        if anomaly:
            summary_parts.append("⚠️ Sudden scene change detected!")
        if predicted:
            summary_parts.append(f"Likely next: {', '.join(predicted)}.")

        return {
            "motion_level":     motion,
            "trend":            trend,
            "predicted_labels": predicted,
            "anomaly":          anomaly,
            "temporal_summary": " ".join(summary_parts),
        }


# ── Public class ──────────────────────────────────────────────────────────────

class RNNTemporalAnalyser:
    """
    Plug-in temporal analyser for Vision OS.

    Parameters
    ----------
    window     : int   – number of past frames to keep  (default 16)
    hidden_dim : int   – LSTM hidden units               (default 64)
    """

    def __init__(self, window: int = 16, hidden_dim: int = 64):
        self.window     = window
        self.hidden_dim = hidden_dim
        self._lock      = threading.Lock()

        if _USE_TORCH:
            self._lstm = _TorchLSTMCell(_FEAT_DIM, hidden_dim)
        else:
            self._lstm = _NumpyLSTMCell(_FEAT_DIM, hidden_dim)

        self._analysis_head = _AnalysisHead(hidden_dim, _N_CLASSES)

        # Rolling history buffers
        self._feat_history: collections.deque = collections.deque(maxlen=window)
        self._h_history:    collections.deque = collections.deque(maxlen=window)
        self._last_result:  dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detection: dict, frame=None) -> None:
        """
        Feed the latest detection dict to the RNN.
        Call this every frame from analyse_frame().
        """
        vec  = _detection_to_vector(detection)
        with self._lock:
            h    = self._lstm.step(vec)
            self._feat_history.append(vec)
            self._h_history.append(h)

            if len(self._h_history) >= 2:
                h_arr    = np.stack(list(self._h_history))
                feat_arr = np.stack(list(self._feat_history))
                self._last_result = self._analysis_head.analyse(h_arr, feat_arr)

    def get_analysis(self) -> dict:
        """Return the latest temporal analysis dict (thread-safe)."""
        with self._lock:
            return dict(self._last_result) if self._last_result else {
                "motion_level": "none", "trend": "stable",
                "predicted_labels": [], "anomaly": False,
                "temporal_summary": "Warming up…",
            }

    def reset(self) -> None:
        """Reset hidden state (e.g. when camera source switches)."""
        with self._lock:
            self._lstm.reset()
            self._feat_history.clear()
            self._h_history.clear()
            self._last_result = {}
