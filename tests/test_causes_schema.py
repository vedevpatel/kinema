"""causes.json schema validation: the cause map can never silently drift
from the engine it interprets. Every check id, severity and docs section
it cites must exist; a stale entry would produce false 'no cause found'
answers, which is exactly what this file exists to prevent."""

from __future__ import annotations

import re
from pathlib import Path

from kinema.checks import ALL_CHECKS, Severity
from kinema.diagnose import CAUSES

_DOCS = Path(__file__).resolve().parent.parent / "docs" / "checks.md"


def _docs_sections() -> set[str]:
    return set(re.findall(r"^## (.+)$", _DOCS.read_text(), flags=re.M))


def test_every_cited_check_exists():
    known = {c.id for c in ALL_CHECKS}
    for cause in CAUSES["causes"]:
        assert cause["indicates"]["check_id"] in known, cause["id"]
    for name, cluster in CAUSES["clusters"].items():
        for cid in cluster["checks"]:
            assert cid in known, f"cluster {name}: {cid}"


def test_every_cited_severity_and_confidence_is_valid():
    for cause in CAUSES["causes"]:
        Severity[cause["indicates"]["min_severity"]]  # raises on a typo
        assert cause["indicates"]["min_severity"] in cause["confidence"], cause["id"]
        for conf in cause["confidence"].values():
            assert conf in ("high", "medium", "low"), cause["id"]


def test_every_cited_docs_section_exists():
    sections = _docs_sections()
    for cause in CAUSES["causes"]:
        assert cause["docs_section"] in sections, cause["id"]


def test_every_cause_has_the_answer_fields():
    for cause in CAUSES["causes"]:
        for key in ("id", "cluster", "title", "explanation", "next_step"):
            assert cause.get(key), f"{cause.get('id')}: missing {key}"
        assert cause["cluster"] in CAUSES["clusters"], cause["id"]
        # explanations format with policy=... ; a {policy} placeholder is optional


def test_clusters_have_keywords_and_checks():
    for name, cluster in CAUSES["clusters"].items():
        assert cluster["keywords"], name
        assert cluster["checks"], name


def test_message_contains_needles_exist_in_check_source():
    """Review finding 14: a cause whose indicates.message_contains matches
    nothing the check can emit silently demotes its findings to
    'outside coverage'. Pin every needle against the check's own source so
    rewording a message breaks the build instead of the diagnosis."""
    import inspect

    by_id = {c.id: c for c in ALL_CHECKS}
    for cause in CAUSES["causes"]:
        needle = cause["indicates"].get("message_contains")
        if not needle:
            continue
        module = inspect.getmodule(type(by_id[cause["indicates"]["check_id"]]))
        source = inspect.getsource(module)
        assert needle in source, (
            f"{cause['id']}: message_contains {needle!r} appears in no message "
            f"its check can emit"
        )
