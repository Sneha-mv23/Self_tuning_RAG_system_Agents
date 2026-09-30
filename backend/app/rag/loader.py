import re
from dataclasses import dataclass
from pathlib import Path

# Lines like {* ../../docs_src/x.py *} are code-include markers, not content.
_INCLUDE_LINE = re.compile(r"^\s*\{[*!].*[*!]\}\s*$")
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


@dataclass(frozen=True)
class Document:
    doc_id: str  # path relative to the corpus dir, e.g. "tutorial/first-steps.md"
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = _FRONT_MATTER.sub("", text)
    lines = [ln for ln in text.split("\n") if not _INCLUDE_LINE.match(ln)]
    return "\n".join(lines).strip() + "\n"


def load_corpus(corpus_dir: str = "data/corpus") -> list[Document]:
    root = Path(corpus_dir)
    docs: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        if path.name == "SOURCE.md":
            continue
        text = clean_text(path.read_text(encoding="utf-8"))
        if len(text.strip()) < 50:
            continue  # skip near-empty pages
        docs.append(Document(doc_id=path.relative_to(root).as_posix(), text=text))
    return docs