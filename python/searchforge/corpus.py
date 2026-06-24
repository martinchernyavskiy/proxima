"""Corpus loading. Default corpus is Wikipedia (via HuggingFace `datasets`).

A corpus is a list of `Doc`s with a human-readable title/snippet for the demo
and an `embed_text` field that is what we actually feed the embedder.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


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
            text=text[:snippet_chars],
            url=row.get("url", ""),
        ))
        if len(docs) >= limit:
            break
    return docs
