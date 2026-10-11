"""Tests for the financial-sentiment module.

The transformer tests build a tiny, randomly initialised BERT and a word-piece tokenizer
from the training sentences, so they run offline in seconds and exercise the real
PyTorch training loop, early stopping and save/load without downloading a model.
"""
import numpy as np
import pytest

from compliance_ai.config import ROOT
from compliance_ai.nlp.data import load_phrasebank, split
from compliance_ai.nlp.models import EmbeddingClassifier, FineTunedTransformer, TfidfBaseline

LABELS = ["negative", "neutral", "positive"]
DATA = ROOT / "data/finance_text/Sentences_75Agree.txt"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="Financial PhraseBank not present")


@pytest.fixture(scope="module")
def splits():
    return split(load_phrasebank(), seed=42)


def tiny_bert(texts, with_head=True):
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertForSequenceClassification, BertModel, PreTrainedTokenizerFast

    torch.manual_seed(0)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(list(texts), trainers.WordPieceTrainer(
        vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    cfg = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, num_labels=3)
    model = BertForSequenceClassification(cfg) if with_head else BertModel(cfg)
    return model, tokenizer


def test_phrasebank_loads(splits):
    df = load_phrasebank()
    assert len(df) > 3400 and set(df.label) == set(LABELS)
    assert not df.text.duplicated().any()
    train, val, test = splits
    assert not set(train.text) & set(test.text)          # no leakage between splits
    assert abs(test.label.eq("neutral").mean() - df.label.eq("neutral").mean()) < 0.02


def test_baseline_beats_majority_class(splits, tmp_path):
    train, _, test = splits
    model = TfidfBaseline(LABELS).fit(train.text, train.label)
    pred = model.predict_proba(test.text).argmax(1)
    acc = np.mean([LABELS[i] for i in pred] == test.label.to_numpy())
    assert acc > test.label.value_counts(normalize=True).max() + 0.1
    model.save(tmp_path)
    again = TfidfBaseline.load(tmp_path)
    assert np.allclose(again.predict_proba(test.text[:5]), model.predict_proba(test.text[:5]))


def test_fine_tuning_loop_learns_and_round_trips(splits, tmp_path):
    train, val, _ = splits
    sub = train.sample(400, random_state=0)
    model, tokenizer = tiny_bert(train.text)
    ft = FineTunedTransformer(LABELS, "tiny-bert", max_length=48, epochs=3, batch_size=32,
                              learning_rate=5e-4, model=model, tokenizer=tokenizer)
    ft.fit(sub.text, sub.label, val.text[:100], val.label[:100])
    losses = [h["train_loss"] for h in ft.history]
    assert losses[-1] < losses[0]                          # it actually learns
    assert all("val_macro_f1" in h for h in ft.history)
    probs = ft.predict_proba(val.text[:10])
    assert probs.shape == (10, 3) and np.allclose(probs.sum(1), 1, atol=1e-5)
    ft.save(tmp_path)
    again = FineTunedTransformer.load(tmp_path)
    assert np.allclose(again.predict_proba(val.text[:10]), probs, atol=1e-5)


def test_frozen_embeddings_pipeline(splits):
    train, _, test = splits
    encoder, tokenizer = tiny_bert(train.text, with_head=False)
    clf = EmbeddingClassifier(LABELS, "tiny-bert", max_length=48, model=encoder, tokenizer=tokenizer)
    clf.fit(train.text[:200], train.label[:200])
    emb = clf.embed(test.text[:4])
    assert emb.shape == (4, 32) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
    assert clf.predict_proba(test.text[:4]).shape == (4, 3)
