"""
Loads disease cards from data/cards/*.yaml.

A "disease card" is the patient-simulator data for one disease — it tells
the env what the patient says when asked a question, what an exam shows,
what a test returns, and which combination of evidence is the minimum
required to declare the diagnosis confirmed.

Cards are kept as plain dicts (not dataclasses) on purpose:
  - the schemas are nested + fluid; converting the whole thing into typed
    dataclasses adds boilerplate without buying type safety where it counts
  - the env reads cards via a few well-defined accessor helpers, and those
    accessors are where the dict shape is interpreted
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Card loading
# ---------------------------------------------------------------------------

def load_cards(cards_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Read every YAML file under cards_dir into a {disease_id: card_dict} map.

    The card's `id` field is the source of truth for the key — filename is
    just convention.
    """
    root = Path(cards_dir)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.yaml")):
        card = yaml.safe_load(path.read_text())
        out[card["id"]] = card
    return out


# ---------------------------------------------------------------------------
# Variant selection
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    """One concrete patient drawn from a card. Fields come straight from the
    chosen entry of card.presentation_variants[]."""

    variant_id: str
    variant_type: str       # "textbook" | "acute_severe" | "atypical"
    age: int
    sex: str
    chief_complaint: str
    patient_card: str       # one-line narrative shown at intake


def select_variant(card: dict[str, Any], rng: random.Random) -> Variant:
    """Pick one of the card's 3 variants uniformly at random.

    During training we may want to bias toward textbook early in the
    curriculum and atypical late — that policy belongs to the patient
    sampler that wraps the env, not the env itself. Here we just pick.
    """
    variants = card.get("presentation_variants", [])
    if not variants:
        raise ValueError(f"card {card.get('id')} has no presentation_variants")
    chosen = rng.choice(variants)
    return Variant(
        variant_id=chosen["variant_id"],
        variant_type=chosen["variant_type"],
        age=int(chosen["age"]),
        sex=chosen["sex"],
        chief_complaint=chosen["chief_complaint"],
        patient_card=chosen["patient_card"],
    )
