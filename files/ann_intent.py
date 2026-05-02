"""
ann_intent.py  —  Vision OS ANN Intent Classifier
==================================================
A feedforward Artificial Neural Network that replaces the hard-coded keyword
chain in the /chat route with a learned (or rapidly-trained) classifier.

Architecture
------------
    Input  : bag-of-words (256-dim) of question tokens  +  scene context
             (object count, obstacle level, currency flag, colour hue)
             → total 260-dim input vector
    Hidden : Layer-1  128 ReLU  →  Dropout(0.3)
             Layer-2   64 ReLU
    Output :  8 softmax classes

Intent classes
--------------
    0  CURRENCY     – money/cash/rupee questions
    1  COLOUR       – colour/shade questions
    2  PEOPLE       – people/person questions
    3  OBSTACLE     – safety/path/walk questions
    4  TEXT_READ    – read/sign/text questions
    5  DIRECTION    – where/left/right questions
    6  OBJECT_QUERY – what/object/see questions  (new vs rule-based)
    7  GENERAL      – everything else

Training strategy
-----------------
The network ships with *synthetic* pre-initialised weights that encode the
keyword priors (so it works without any training data), AND supports
incremental online training via `train_one(question, label_idx)` so it
improves from real interactions.

If scikit-learn is available the weights are kept in a proper
MLPClassifier and the synthetic init is replaced by a warm-started fit
on 200 synthetic examples.  Otherwise the pure-numpy forward pass is used.

Usage
-----
    from ann_intent import ANNIntentClassifier

    ann = ANNIntentClassifier()

    intent, confidence = ann.predict("what colour is it?", detection_dict)
    # → ("COLOUR", 0.91)

    # Optional: improve from feedback
    ann.train_one("what colour is it?", 1)   # label 1 = COLOUR
"""

import numpy as np
import threading
import re
from typing import Tuple, Dict, List, Optional

# ── Intent registry ──────────────────────────────────────────────────────────
INTENT_LABELS = [
    "CURRENCY",     # 0
    "COLOUR",       # 1
    "PEOPLE",       # 2
    "OBSTACLE",     # 3
    "TEXT_READ",    # 4
    "DIRECTION",    # 5
    "OBJECT_QUERY", # 6
    "GENERAL",      # 7
]
N_INTENTS = len(INTENT_LABELS)

# ── Vocabulary ────────────────────────────────────────────────────────────────
# 256-word bag-of-words vocabulary (covers common questions for this domain)
_VOCAB_SEEDS = [
    # CURRENCY
    "money","cash","currency","rupee","note","price","pay","coin","wallet",
    "denomination","bill","banknote","thousand","hundred","fifty","twenty",
    # COLOUR
    "color","colour","shade","hue","tint","dark","light","bright","pale",
    "red","green","blue","yellow","orange","purple","pink","white","black","grey",
    # PEOPLE
    "people","person","human","someone","anybody","man","woman","child",
    "crowd","face","standing","walking","sitting",
    # OBSTACLE / SAFETY
    "safe","walk","obstacle","block","path","clear","danger","warning","ahead",
    "distance","near","close","wall","door","step","stairs","floor",
    # TEXT / OCR
    "read","text","sign","write","written","label","number","letter","word",
    "board","display","screen","poster","name",
    # DIRECTION
    "where","direction","left","right","front","behind","above","below",
    "beside","next","around","corner","position","locate",
    # OBJECT QUERY
    "what","see","detect","find","show","object","thing","item","identify",
    "around","there","visible","spot","notice","look",
    # GENERAL
    "how","why","when","tell","describe","explain","help","can","is","are",
    "do","does","will","should","could","would","please","thanks","okay",
]

# Deduplicate and cap at 256
_VOCAB_LIST = list(dict.fromkeys(_VOCAB_SEEDS))[:256]
_VOCAB_SIZE  = len(_VOCAB_LIST)
_VOCAB_IDX   = {w: i for i, w in enumerate(_VOCAB_LIST)}

# Context features appended after BoW (4 extra dims → total = _VOCAB_SIZE + 4)
_INPUT_DIM   = _VOCAB_SIZE + 4
_H1_DIM      = 128
_H2_DIM      = 64


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _text_to_bow(text: str) -> np.ndarray:
    vec = np.zeros(_VOCAB_SIZE, dtype=np.float32)
    for tok in _tokenise(text):
        idx = _VOCAB_IDX.get(tok)
        if idx is not None:
            vec[idx] += 1.0
    norm = vec.max()
    if norm > 0:
        vec /= norm
    return vec


def _context_features(detection: dict) -> np.ndarray:
    """4-dim scene context vector from the latest detection."""
    feat = np.zeros(4, dtype=np.float32)
    # [0] object count (norm)
    feat[0] = min(len(detection.get("objects", [])), 20) / 20.0
    # [1] obstacle level
    lvl_map = {"SAFE": 0.0, "WARN": 0.5, "WARNING": 0.5, "DANGER": 1.0}
    feat[1] = lvl_map.get(str(detection.get("obstacle", {}).get("level", "SAFE")).upper(), 0.0)
    # [2] currency present
    feat[2] = 1.0 if detection.get("currency") else 0.0
    # [3] dominant-colour hue
    cols = detection.get("dominant_colours", [])
    if cols:
        hex_col = cols[0].get("hex", "#808080")
        try:
            r = int(hex_col[1:3], 16) / 255.0
            g = int(hex_col[3:5], 16) / 255.0
            b = int(hex_col[5:7], 16) / 255.0
            import colorsys
            h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
            feat[3] = float(h)
        except Exception:
            feat[3] = 0.5
    return feat


def _build_input(question: str, detection: dict) -> np.ndarray:
    return np.concatenate([_text_to_bow(question), _context_features(detection)])


# ── Numpy forward pass ────────────────────────────────────────────────────────

def _relu(x):   return np.maximum(0.0, x)
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class _NumpyANN:
    """2-hidden-layer MLP with synthetic keyword-prior weights."""

    def __init__(self):
        rng = np.random.default_rng(7)

        # Layer 1: _INPUT_DIM → _H1_DIM
        self.W1 = rng.normal(0, 0.05, (_H1_DIM, _INPUT_DIM)).astype(np.float32)
        self.b1 = np.zeros(_H1_DIM, dtype=np.float32)

        # Layer 2: _H1_DIM → _H2_DIM
        self.W2 = rng.normal(0, 0.1, (_H2_DIM, _H1_DIM)).astype(np.float32)
        self.b2 = np.zeros(_H2_DIM, dtype=np.float32)

        # Output layer: _H2_DIM → N_INTENTS
        self.W3 = rng.normal(0, 0.1, (N_INTENTS, _H2_DIM)).astype(np.float32)
        self.b3 = np.zeros(N_INTENTS, dtype=np.float32)

        # ── Encode keyword priors directly into W1 ────────────────────────────
        # Each intent gets a boosted weight for its seed vocabulary in the
        # first-layer neurons, so the network has a strong prior even without
        # any supervised training.
        _PRIOR_WORDS: Dict[int, List[str]] = {
            0: ["money","cash","currency","rupee","note","denomination","bill"],
            1: ["color","colour","shade","hue","red","green","blue","yellow"],
            2: ["people","person","human","someone","man","woman","crowd"],
            3: ["safe","walk","obstacle","block","path","danger","distance"],
            4: ["read","text","sign","write","word","board","display"],
            5: ["where","direction","left","right","front","corner","locate"],
            6: ["what","see","detect","find","object","thing","identify"],
            7: ["how","why","when","help","can","please","thanks"],
        }
        for intent_idx, words in _PRIOR_WORDS.items():
            for word in words:
                vocab_idx = _VOCAB_IDX.get(word)
                if vocab_idx is not None:
                    # Boost first _H1_DIM//8 neurons assigned to this intent
                    start = intent_idx * (_H1_DIM // N_INTENTS)
                    end   = start + (_H1_DIM // N_INTENTS)
                    self.W1[start:end, vocab_idx] += 0.8

        # Wire W3 so each intent's hidden block connects strongly to its output
        for intent_idx in range(N_INTENTS):
            start = intent_idx * (_H2_DIM // N_INTENTS)
            end   = start + (_H2_DIM // N_INTENTS)
            self.W3[intent_idx, start:end] += 1.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities shape (N_INTENTS,)."""
        h1 = _relu(self.W1 @ x + self.b1)
        h2 = _relu(self.W2 @ h1 + self.b2)
        return _softmax(self.W3 @ h2 + self.b3)

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        probs = self.forward(x)
        idx   = int(np.argmax(probs))
        return idx, float(probs[idx])


# ── Optional scikit-learn path ────────────────────────────────────────────────
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing  import LabelEncoder
    _SK_AVAILABLE = True
    print("[ANN] ✅ scikit-learn detected — using MLPClassifier")
except ImportError:
    _SK_AVAILABLE = False
    print("[ANN] ℹ️  scikit-learn not found — using numpy ANN (fully functional)")


def _make_synthetic_samples(n_per_class: int = 30):
    """Generate synthetic (X, y) training pairs from keyword priors."""
    _TEMPLATES: Dict[int, List[str]] = {
        0: ["what money is this","detect currency","read rupee note","identify cash",
            "how much is this","what denomination","is this a note","what banknote"],
        1: ["what color is it","what colour do you see","what shade is this",
            "tell me the colour","what hue","is it dark or light","describe colour"],
        2: ["how many people","is there anyone","is someone there","how many persons",
            "any humans","detect person","is there a crowd","who is nearby"],
        3: ["is it safe to walk","any obstacles ahead","is path clear",
            "what is in front","danger ahead","how far","is there a wall"],
        4: ["read the text","what does it say","any signs","read the board",
            "what is written","detect text","read label","any words"],
        5: ["where is it","what is on the left","what is on the right",
            "what direction","where is the object","locate it","which side"],
        6: ["what do you see","what objects are there","detect objects",
            "what is around me","identify items","what things","spot anything"],
        7: ["help me","how are you","can you help","okay","thanks","please",
            "what can you do","tell me something"],
    }
    rng = np.random.default_rng(42)
    X, y = [], []
    for label_idx, templates in _TEMPLATES.items():
        for _ in range(n_per_class):
            tmpl = templates[rng.integers(len(templates))]
            # Light augmentation: randomly drop words
            words = tmpl.split()
            if len(words) > 2:
                keep = rng.random(len(words)) > 0.2
                words = [w for w, k in zip(words, keep) if k] or words
            q = " ".join(words)
            dummy_det: dict = {}
            x = _build_input(q, dummy_det)
            X.append(x)
            y.append(label_idx)
    return np.array(X), np.array(y)


# ── Public class ──────────────────────────────────────────────────────────────

class ANNIntentClassifier:
    """
    Drop-in intent classifier for Vision OS /chat route.

    Parameters
    ----------
    use_sklearn : bool  – use MLPClassifier if available (default: auto)
    """

    def __init__(self, use_sklearn: bool = True):
        self._lock = threading.Lock()
        self._use_sklearn = use_sklearn and _SK_AVAILABLE

        if self._use_sklearn:
            self._clf = MLPClassifier(
                hidden_layer_sizes=(_H1_DIM, _H2_DIM),
                activation="relu",
                solver="adam",
                max_iter=300,
                random_state=42,
                warm_start=True,
            )
            X, y = _make_synthetic_samples(n_per_class=30)
            self._clf.fit(X, y)
            self._online_X: List[np.ndarray] = []
            self._online_y: List[int]         = []
            self._retrain_every = 20   # retrain after every N online samples
            print(f"[ANN] MLPClassifier trained on {len(y)} synthetic samples")
        else:
            self._mlp = _NumpyANN()
            print("[ANN] numpy ANN ready (keyword-prior weights)")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        question:  str,
        detection: Optional[dict] = None,
    ) -> Tuple[str, float]:
        """
        Classify the user's question.

        Returns
        -------
        (intent_label, confidence)   e.g.  ("COLOUR", 0.91)
        """
        if detection is None:
            detection = {}
        x = _build_input(question, detection)

        with self._lock:
            if self._use_sklearn:
                probs   = self._clf.predict_proba(x.reshape(1, -1))[0]
                idx     = int(np.argmax(probs))
                conf    = float(probs[idx])
            else:
                idx, conf = self._mlp.predict(x)

        return INTENT_LABELS[idx], conf

    # ── Online training ───────────────────────────────────────────────────────

    def train_one(
        self,
        question:  str,
        label_idx: int,
        detection: Optional[dict] = None,
    ) -> None:
        """
        Add one labelled example and, if sklearn path, periodically retrain.

        Parameters
        ----------
        question  : raw user question string
        label_idx : integer index into INTENT_LABELS
        detection : optional current detection dict for context features
        """
        if detection is None:
            detection = {}
        if not (0 <= label_idx < N_INTENTS):
            return
        x = _build_input(question, detection)

        with self._lock:
            if self._use_sklearn:
                self._online_X.append(x)
                self._online_y.append(label_idx)
                if len(self._online_X) % self._retrain_every == 0:
                    # Merge with synthetic baseline to prevent catastrophic forgetting
                    X_base, y_base = _make_synthetic_samples(n_per_class=10)
                    X_all = np.vstack([X_base] + [v.reshape(1,-1) for v in self._online_X])
                    y_all = np.concatenate([y_base, self._online_y])
                    self._clf.fit(X_all, y_all)
                    print(f"[ANN] Retrained on {len(y_all)} samples ({len(self._online_X)} real)")
            # numpy path: no online training (weights are fixed priors)

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_intent_labels(self) -> List[str]:
        return list(INTENT_LABELS)

    def label_to_idx(self, label: str) -> int:
        return INTENT_LABELS.index(label) if label in INTENT_LABELS else 7
