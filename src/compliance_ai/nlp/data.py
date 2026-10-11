"""Load and split Financial PhraseBank (sentence@label, ISO-8859-1)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from compliance_ai.config import load_config, resolve


def load_phrasebank(path: str | None = None) -> pd.DataFrame:
    cfg = load_config()["nlp"]
    rows = []
    with open(resolve(path or cfg["data_path"]), encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            text, _, label = line.rpartition("@")
            rows.append({"text": text.strip(), "label": label.strip()})
    df = pd.DataFrame(rows)
    unknown = set(df.label) - set(cfg["labels"])
    if unknown:
        raise ValueError(f"unexpected labels: {unknown}")
    # The release contains a few exact duplicate sentences; keep one so they cannot leak across splits.
    return df.drop_duplicates("text").reset_index(drop=True)


def split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split."""
    cfg = load_config()["nlp"]
    train_val, test = train_test_split(df, test_size=cfg["test_size"], stratify=df.label, random_state=seed)
    train, val = train_test_split(train_val, test_size=cfg["val_size"], stratify=train_val.label,
                                  random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
