"""Public surface of the env package.

Typical usage:

    from clinical_rl.env import ClinicalEnv, Action, ActionType, load_catalogs, load_cards

    catalogs = load_catalogs("data")
    cards = load_cards("data/cards")
    env = ClinicalEnv(catalogs, cards)

    obs = env.reset(disease="ectopic_pregnancy", seed=42)
    obs = env.step(Action(ActionType.INTERVIEW, "pain_location"))
"""

from .cards import Variant, load_cards, select_variant
from .catalogs import Catalogs, Disease, Exam, Question, Test, load_catalogs
from .env import ClinicalEnv
from .state import (
    Action,
    ActionType,
    Episode,
    HiddenState,
    LogEntry,
    Observation,
    PendingTest,
)

__all__ = [
    # env
    "ClinicalEnv",
    # state
    "Action",
    "ActionType",
    "Episode",
    "HiddenState",
    "LogEntry",
    "Observation",
    "PendingTest",
    # catalogs
    "Catalogs",
    "Disease",
    "Question",
    "Exam",
    "Test",
    "load_catalogs",
    # cards
    "Variant",
    "load_cards",
    "select_variant",
]
