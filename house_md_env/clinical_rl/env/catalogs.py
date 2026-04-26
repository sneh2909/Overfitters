"""
Loads the four catalog files from data/ and exposes O(1) lookup by id.

The catalogs are FROZEN — `data/{diseases,questions,exams,tests}.yaml` define
the action vocabulary the agent emits. Any rename here invalidates every
disease card under data/cards/.

A single `Catalogs` object bundles all four; the env constructor takes one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Per-catalog row dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Disease:
    id: str
    name: str
    family: str
    severity: str            # "stable" | "urgent" | "critical"
    deterioration_rate: float
    suggested_red_herring: Optional[str]
    sex_allowed: tuple[str, ...]    # ("female",) or ("male","female")
    age_min: int
    age_max: int


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str


@dataclass(frozen=True)
class Exam:
    id: str
    category: str
    text: str
    cost: int                # USD
    time_min: int            # minutes added to clock


@dataclass(frozen=True)
class Test:
    id: str
    name: str
    category: str
    cost: int                # USD
    turnaround_steps: int    # 0 = same step (POC), 1-3 = delayed


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class Catalogs:
    """All four catalogs together. Lookups are dict-based for O(1)."""

    diseases: dict[str, Disease]
    questions: dict[str, Question]
    exams: dict[str, Exam]
    tests: dict[str, Test]

    # ---- convenience predicates the env uses every step ----

    def is_valid_question(self, qid: str) -> bool:
        return qid in self.questions

    def is_valid_exam(self, eid: str) -> bool:
        return eid in self.exams

    def is_valid_test(self, tid: str) -> bool:
        return tid in self.tests

    def is_valid_disease(self, did: str) -> bool:
        return did in self.diseases


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_catalogs(data_dir: str | Path) -> Catalogs:
    """Read all four YAML files and build a Catalogs bundle.

    Parameters
    ----------
    data_dir
        Path to the `data/` directory containing diseases.yaml, questions.yaml,
        exams.yaml, tests.yaml.
    """
    root = Path(data_dir)

    # --- diseases.yaml ---
    # Schema: { version, count, diseases: [ {id, name, family, severity,
    #          deterioration_rate, suggested_red_herring, constraints: {sex_allowed, age_range}} ] }
    raw_d = yaml.safe_load((root / "diseases.yaml").read_text())
    diseases: dict[str, Disease] = {}
    for d in raw_d["diseases"]:
        constraints = d.get("constraints", {})
        age_range = constraints.get("age_range", [0, 120])
        diseases[d["id"]] = Disease(
            id=d["id"],
            name=d["name"],
            family=d["family"],
            severity=d["severity"],
            deterioration_rate=float(d.get("deterioration_rate", 0.0)),
            suggested_red_herring=d.get("suggested_red_herring"),
            sex_allowed=tuple(constraints.get("sex_allowed", ["male", "female"])),
            age_min=int(age_range[0]),
            age_max=int(age_range[1]),
        )

    # --- questions.yaml ---
    raw_q = yaml.safe_load((root / "questions.yaml").read_text())
    questions = {
        q["id"]: Question(id=q["id"], category=q["category"], text=q["text"])
        for q in raw_q["questions"]
    }

    # --- exams.yaml ---
    raw_e = yaml.safe_load((root / "exams.yaml").read_text())
    exams = {
        e["id"]: Exam(
            id=e["id"],
            category=e["category"],
            text=e["text"],
            cost=int(e.get("cost", 0)),
            time_min=int(e.get("time_min", 0)),
        )
        for e in raw_e["exams"]
    }

    # --- tests.yaml ---
    raw_t = yaml.safe_load((root / "tests.yaml").read_text())
    tests = {
        t["id"]: Test(
            id=t["id"],
            name=t["name"],
            category=t["category"],
            cost=int(t["cost"]),
            turnaround_steps=int(t["turnaround_steps"]),
        )
        for t in raw_t["tests"]
    }

    # Sanity: catalog counts match the contract documented in env_contract memory.
    assert len(diseases) == 15, f"expected 15 diseases, got {len(diseases)}"
    assert len(questions) == 25, f"expected 25 questions, got {len(questions)}"
    assert len(exams) == 15, f"expected 15 exams, got {len(exams)}"
    assert len(tests) == 35, f"expected 35 tests, got {len(tests)}"

    return Catalogs(diseases=diseases, questions=questions, exams=exams, tests=tests)
