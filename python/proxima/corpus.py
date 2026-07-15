"""Corpus loading. Default corpus is Wikipedia (via HuggingFace `datasets`).

A corpus is a list of `Doc`s with a human-readable title/snippet for the demo
and an `embed_text` field that is what we actually feed the embedder.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


def trim_snippet(text: str, max_chars: int = 500, *, already_cut: bool = False) -> str:
    """Trim `text` to at most `max_chars`. Prefers a clean word boundary over
    a mid-word cut, and only appends an ellipsis when the cut doesn't already
    land on a real sentence boundary — a truncation that happens to land right
    after a "." reads as a complete sentence and looks better left alone than
    forced into "...species…".

    `already_cut=True` is for cleaning up corpora built before this function
    existed, whose snippets were hard-sliced at exactly `max_chars` and may end
    mid-word: `len(text) == max_chars` alone can't tell "this is a complete
    string that happens to be exactly max_chars long" (leave it alone) apart
    from "this was hard-truncated at exactly max_chars" (needs cleanup) — the
    two calls disambiguate by declaring which situation they're in, rather
    than guessing from length alone.
    """
    max_chars = max(max_chars, 0)
    if len(text) < max_chars or (len(text) == max_chars and not already_cut):
        return text
    cut = text[:max_chars].rstrip()
    if not cut or cut[-1] in ".!?":
        return cut  # already ends on a real sentence boundary — leave it be
    if cut[-1] not in " ,;:\"')]}":
        # Cut lands mid-word (or mid-clause with no punctuation): back up to
        # the last full word before marking the truncation.
        last_space = cut.rfind(" ")
        if last_space > max_chars * 0.6:  # keep most of the budget
            cut = cut[:last_space]
    return cut.rstrip(" ,;:") + "…"


@dataclass
class Doc:
    id: int
    title: str
    text: str   # display snippet
    url: str

    @property
    def embed_text(self) -> str:
        # Title + lead snippet embeds noticeably better than the body alone:
        # the title supplies the topic, the snippet the detail.
        return f"{self.title}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def load_wikipedia(limit: int = 100_000, config: str = "20231101.simple",
                   snippet_chars: int = 500, min_chars: int = 200) -> list[Doc]:
    """Stream Wikipedia articles and return up to `limit` `Doc`s.

    Uses the `wikimedia/wikipedia` dataset (clean parquet). `20231101.simple` is
    Simple English Wikipedia (~240k articles): a real, recruiter-legible corpus
    that is small enough to download and embed quickly. Pass `20231101.en` for
    full English Wikipedia at the scale milestone.
    """
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    docs: list[Doc] = []
    for row in ds:
        text = " ".join(row["text"].split())  # collapse whitespace/newlines
        if len(text) < min_chars:
            continue  # skip stubs/disambiguation-like pages
        docs.append(Doc(
            id=len(docs),
            title=row["title"],
            text=trim_snippet(text, snippet_chars),
            url=row.get("url", ""),
        ))
        if len(docs) >= limit:
            break
    return docs
