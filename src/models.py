"""Three intent classifiers with the same interface: fit(texts, labels) and predict_proba(texts) -> (probs, classes).

1. TF-IDF (word + character n-grams) + logistic regression  - classic baseline, trains in seconds
2. Sentence embeddings (bge-small) + logistic regression     - frozen pretrained encoder, no fine-tuning
3. Fine-tuned DistilBERT                                      - full transformer fine-tuning
"""
import json

import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

SEED = 42


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TfidfLR:
    name = "TF-IDF + LogReg"

    def fit(self, texts, labels):
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        # character n-grams make the model robust to typos and word forms ("transfered", "transferring")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        X = hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        self.clf = LogisticRegression(C=20, max_iter=3000, random_state=SEED).fit(X, labels)
        self.classes = self.clf.classes_
        return self

    def predict_proba(self, texts):
        X = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return self.clf.predict_proba(X), self.classes


class EmbeddingLR:
    name = "bge-small embeddings + LogReg"

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name, device=str(get_device()))

    def _embed(self, texts):
        return self.encoder.encode(list(texts), batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    def fit(self, texts, labels):
        self.clf = LogisticRegression(C=10, max_iter=3000, random_state=SEED).fit(self._embed(texts), labels)
        self.classes = self.clf.classes_
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._embed(texts)), self.classes


class DistilBert:
    name = "Fine-tuned DistilBERT"

    def __init__(self, model_name="distilbert-base-uncased", epochs=4, lr=5e-5, batch_size=32, max_len=32):
        self.model_name, self.epochs, self.lr, self.batch_size, self.max_len = model_name, epochs, lr, batch_size, max_len
        self.device = get_device()

    def _batches(self, texts, labels=None, shuffle=False):
        idx = list(range(len(texts)))
        loader = DataLoader(idx, batch_size=self.batch_size, shuffle=shuffle,
                            generator=torch.Generator().manual_seed(SEED))
        for b in loader:
            enc = self.tok([texts[i] for i in b], padding=True, truncation=True, max_length=self.max_len,
                           return_tensors="pt").to(self.device)
            y = None if labels is None else torch.tensor([labels[i] for i in b], device=self.device)
            yield enc, y

    def fit(self, texts, labels):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
        torch.manual_seed(SEED)
        texts = list(texts)
        self.classes = np.array(sorted(set(labels)))
        y = [int(np.searchsorted(self.classes, label)) for label in labels]
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.classes)).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * ((len(texts) + self.batch_size - 1) // self.batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            for enc, yb in self._batches(texts, y, shuffle=True):
                loss = self.model(**enc, labels=yb).loss
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad()
                total += loss.item()
            print(f"  epoch {epoch + 1}: mean loss {total / (steps / self.epochs):.4f}")
        return self

    @torch.no_grad()
    def predict_proba(self, texts):
        self.model.eval()
        out = [torch.softmax(self.model(**enc).logits, -1).cpu() for enc, _ in self._batches(list(texts))]
        return torch.cat(out).numpy(), self.classes

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        (path / "classes.json").write_text(json.dumps(self.classes.tolist()))
