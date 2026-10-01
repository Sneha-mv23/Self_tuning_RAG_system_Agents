import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from app.rag.loader import Document

QuestionType = Literal["factual", "paraphrase", "multi_hop", "unanswerable"]

DEFAULT_PATH = "data/golden_set/golden.jsonl"


class Evidence(BaseModel):
    doc_id: str
    quote: str  # exact text copied from the cleaned document
    start: int | None = None  # filled in by resolve_evidence
    end: int | None = None


class GoldenItem(BaseModel):
    id: str
    question: str
    type: QuestionType
    reference_answer: str | None = None
    evidence: list[Evidence] = []

    @model_validator(mode="after")
    def _check_rules(self):
        if self.type == "unanswerable":
            if self.evidence or self.reference_answer is not None:
                raise ValueError(f"{self.id}: unanswerable items must have no evidence and no reference_answer")
        else:
            if not self.evidence:
                raise ValueError(f"{self.id}: answerable items need at least one evidence quote")
            if not self.reference_answer:
                raise ValueError(f"{self.id}: answerable items need a reference_answer")
            if self.type == "multi_hop" and len(self.evidence) < 2:
                raise ValueError(f"{self.id}: multi_hop items need at least two evidence quotes")
        return self


def resolve_evidence(items: list[GoldenItem], docs: list[Document]) -> list[GoldenItem]:
    """Turn each evidence quote into a (start, end) character span."""
    texts = {d.doc_id: d.text for d in docs}
    resolved_items = []
    for item in items:
        resolved = []
        for ev in item.evidence:
            text = texts.get(ev.doc_id)
            if text is None:
                raise ValueError(f"{item.id}: unknown doc_id '{ev.doc_id}'")
            n = text.count(ev.quote)
            if n == 0:
                raise ValueError(f"{item.id}: quote not found in {ev.doc_id}: {ev.quote[:60]!r}")
            if n > 1:
                raise ValueError(f"{item.id}: quote appears {n} times in {ev.doc_id}; use a longer, more specific quote")
            start = text.index(ev.quote)
            resolved.append(ev.model_copy(update={"start": start, "end": start + len(ev.quote)}))
        resolved_items.append(item.model_copy(update={"evidence": resolved}))
    return resolved_items


def load_golden(path: str = DEFAULT_PATH, docs: list[Document] | None = None) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    # utf-8-sig tolerates the invisible BOM that Windows editors sometimes add
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = GoldenItem.model_validate(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path} line {line_no}: {e}") from e
            if item.id in seen:
                raise ValueError(f"duplicate id '{item.id}' at line {line_no}")
            seen.add(item.id)
            items.append(item)
    return resolve_evidence(items, docs) if docs is not None else items