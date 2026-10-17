"""Amounts in EUR, INR or DM.

The model was trained on loan amounts in Deutsche Mark (Statlog German Credit, 1973-75). People enter
and read amounts in their own currency; this module converts them to DM before scoring and back for
display. Rates live in config.yaml (`currency:`).

Converted, not inflation-adjusted: 1,000 DM in 1974 bought much more than 511 EUR buys today, so the
model's view of "a large loan" is a 1970s one. EUR-DM uses the fixed legal rate; INR uses a market
rate on a stated date, and the model is not validated for Indian applicants.
"""
from __future__ import annotations

import re

from compliance_ai.config import load_config

SYMBOLS = {"EUR": "€", "INR": "₹", "DM": "DM"}
NAMES = {"EUR": "Euro", "INR": "Indian rupee", "DM": "Deutsche Mark (training data)"}
MONEY_FEATURES = {"amount", "amount_per_month"}
_DM_AMOUNT = re.compile(r"(\d[\d,]*)\s*DM\b")


def _cfg() -> dict:
    return load_config().get("currency", {})


def per_eur() -> dict[str, float]:
    rates = {"EUR": 1.0, "DM": 1.95583}
    rates.update({k.upper(): float(v) for k, v in _cfg().get("per_eur", {}).items()})
    return rates


def check(code: str | None) -> str:
    """Normalise a currency code; raise ValueError for an unknown one."""
    code = (code or "DM").strip().upper().replace("DEM", "DM")
    if code not in SYMBOLS or code not in per_eur():
        raise ValueError(f"Unknown currency {code!r}; use one of {', '.join(SYMBOLS)}")
    return code


def default() -> str:
    return check(_cfg().get("default", "EUR"))


def dm_per_unit(code: str) -> float:
    """How many DM one unit of `code` is worth (EUR: 1.95583, DM: 1)."""
    rates = per_eur()
    return rates["DM"] / rates[check(code)]


def to_dm(value: float, code: str) -> float:
    return value * dm_per_unit(code)


def from_dm(value: float, code: str) -> float:
    return value / dm_per_unit(code)


def _indian(text: str) -> str:
    """'1001068' -> '10,01,068' (lakh and crore grouping)."""
    head, last3 = text[:-3], text[-3:]
    groups = []
    while len(head) > 2:
        head, groups = head[:-2], [head[-2:], *groups]
    return ",".join([g for g in [head, *groups] if g] + [last3]) if head or groups else last3


def money(value_dm: float, code: str, digits: int = 0) -> str:
    """A DM amount shown in `code`, for example 1,374 DM -> '€703' or '₹74,656'."""
    code = check(code)
    v = from_dm(float(value_dm), code)
    if code == "INR":
        whole, _, frac = f"{abs(v):.{digits}f}".partition(".")
        text = ("-" if v < 0 else "") + _indian(whole) + (f".{frac}" if frac else "")
    else:
        text = f"{v:,.{digits}f}"
    return f"{text} DM" if code == "DM" else f"{SYMBOLS[code]}{text}"


def relabel(text: str, code: str) -> str:
    """Rewrite the DM amounts inside a codebook label, e.g. '... < 100 DM' -> '... < €51'."""
    if check(code) == "DM":
        return text
    return _DM_AMOUNT.sub(lambda m: money(float(m.group(1).replace(",", "")), code), text)


def info(code: str) -> dict:
    """What the pages show about a currency: symbol, rate and an honest note."""
    code = check(code)
    inr, inr_date = per_eur()["INR"], _cfg().get("inr_rate_date", "n/a")
    note = {
        "DM": "Amounts in Deutsche Mark, as in the training data.",
        "EUR": "Converted at the fixed legal rate 1 € = 1.95583 DM. The model learned from 1970s loans, "
               "so amounts are converted, not adjusted for inflation.",
        "INR": f"Converted at 1 € = {inr:g} ₹ (market rate on {inr_date}) "
               "and the fixed 1 € = 1.95583 DM. The model was trained on German applicants and is not "
               "validated for India.",
    }[code]
    return {"code": code, "symbol": SYMBOLS[code], "name": NAMES[code],
            "dm_per_unit": dm_per_unit(code), "note": note}


def all_info() -> list[dict]:
    return [info(c) for c in ("EUR", "INR", "DM")]
