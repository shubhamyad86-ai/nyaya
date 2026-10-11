"""Three financial-sentiment classifiers behind one interface: fit / predict_proba / save / load.

1. TfidfBaseline       - TF-IDF word n-grams + logistic regression (seconds on a laptop).
2. EmbeddingClassifier - a frozen Hugging Face encoder (mean-pooled) + logistic regression.
                         No gradient updates, so it runs on a CPU in minutes.
3. FineTunedTransformer - the whole transformer fine-tuned end to end in PyTorch
                         (AdamW, linear warm-up/decay, early stopping on validation macro-F1).
                         Use a GPU (e.g. free Google Colab); a CPU works but is slow.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


class TfidfBaseline:
    kind = "tfidf_baseline"

    def __init__(self, labels: list[str], seed: int = 42):
        self.labels = labels
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=5.0, class_weight="balanced", max_iter=2000, random_state=seed)),
        ])

    def fit(self, texts, y, val_texts=None, val_y=None):
        self.pipe.fit(list(texts), [self.labels.index(v) for v in y])
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipe.predict_proba(list(texts))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"labels": self.labels, "pipe": self.pipe}, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> TfidfBaseline:
        blob = joblib.load(path / "model.joblib")
        obj = cls(blob["labels"])
        obj.pipe = blob["pipe"]
        return obj


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EmbeddingClassifier:
    kind = "frozen_embeddings"

    def __init__(self, labels: list[str], model_name: str, max_length: int = 64, batch_size: int = 32,
                 seed: int = 42, model=None, tokenizer=None):
        self.labels, self.model_name = labels, model_name
        self.max_length, self.batch_size, self.seed = max_length, batch_size, seed
        self._model, self._tokenizer = model, tokenizer
        self.clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000, random_state=seed)

    def _load_encoder(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(_device()).eval()

    def embed(self, texts) -> np.ndarray:
        import torch

        self._load_encoder()
        texts, out = list(texts), []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self._tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                      max_length=self.max_length, return_tensors="pt").to(_device())
                hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # mean over real tokens
                out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        return np.vstack(out)

    def fit(self, texts, y, val_texts=None, val_y=None):
        _set_seed(self.seed)
        self.clf.fit(self.embed(texts), [self.labels.index(v) for v in y])
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self.embed(texts))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"labels": self.labels, "model_name": self.model_name, "max_length": self.max_length,
                     "clf": self.clf}, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> EmbeddingClassifier:
        blob = joblib.load(path / "model.joblib")
        obj = cls(blob["labels"], blob["model_name"], blob["max_length"])
        obj.clf = blob["clf"]
        return obj


class FineTunedTransformer:
    kind = "fine_tuned_transformer"

    def __init__(self, labels: list[str], model_name: str, max_length: int = 64, epochs: int = 4,
                 batch_size: int = 16, learning_rate: float = 3e-5, seed: int = 42,
                 model=None, tokenizer=None):
        self.labels, self.model_name = labels, model_name
        self.max_length, self.epochs, self.batch_size = max_length, epochs, batch_size
        self.learning_rate, self.seed = learning_rate, seed
        self.model, self.tokenizer = model, tokenizer
        self.history: list[dict] = []

    def _init_model(self):
        if self.model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(self.labels),
                id2label=dict(enumerate(self.labels)), label2id={v: i for i, v in enumerate(self.labels)},
            )
        self.model.to(_device())

    def _batches(self, texts, y=None, shuffle=False):
        import torch

        idx = np.arange(len(texts))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            b = idx[i:i + self.batch_size]
            enc = self.tokenizer([texts[j] for j in b], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            if y is not None:
                enc["labels"] = torch.tensor([y[j] for j in b])
            yield enc.to(_device())

    def fit(self, texts, y, val_texts=None, val_y=None):
        import torch
        from transformers import get_linear_schedule_with_warmup

        _set_seed(self.seed)
        self._init_model()
        texts, y_idx = list(texts), [self.labels.index(v) for v in y]

        # Class weights counter the imbalance (most sentences are neutral).
        counts = np.bincount(y_idx, minlength=len(self.labels))
        weights = torch.tensor(len(y_idx) / (len(self.labels) * np.maximum(counts, 1)),
                               dtype=torch.float, device=_device())
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        optim = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        steps = self.epochs * int(np.ceil(len(texts) / self.batch_size))
        sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)

        best_f1, best_state = -1.0, None
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total = 0.0
            for batch in self._batches(texts, y_idx, shuffle=True):
                labels = batch.pop("labels")
                loss = loss_fn(self.model(**batch).logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                total += loss.item() * len(labels)
            record = {"epoch": epoch, "train_loss": total / len(texts)}
            if val_texts is not None:
                val_pred = self.predict_proba(val_texts).argmax(1)
                record["val_macro_f1"] = f1_score([self.labels.index(v) for v in val_y], val_pred,
                                                  average="macro")
                if record["val_macro_f1"] > best_f1:            # early stopping: keep the best epoch
                    best_f1 = record["val_macro_f1"]
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.history.append(record)
            print(f"  epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "epoch"))
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        import torch

        self.model.eval()
        texts, out = list(texts), []
        with torch.no_grad():
            for batch in self._batches(texts):
                out.append(torch.softmax(self.model(**batch).logits, dim=-1).cpu().numpy())
        return np.vstack(out)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "sentiment_meta.json").write_text(json.dumps({
            "labels": self.labels, "base_model": self.model_name, "max_length": self.max_length,
            "history": self.history}, indent=2))

    @classmethod
    def load(cls, path: Path) -> FineTunedTransformer:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads((path / "sentiment_meta.json").read_text())
        obj = cls(meta["labels"], meta["base_model"], meta["max_length"])
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path).to(_device())
        obj.history = meta.get("history", [])
        return obj


METHODS = {c.kind: c for c in (TfidfBaseline, EmbeddingClassifier, FineTunedTransformer)}
