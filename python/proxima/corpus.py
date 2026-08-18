from __future__ import annotations

from dataclasses import dataclass, asdict


def trim_snippet(text: str, max_chars: int = 500, *, already_cut: bool = False) -> str:
    max_chars = max(max_chars, 0)
    if len(text) < max_chars or (len(text) == max_chars and not already_cut):
        return text
    cut = text[:max_chars].rstrip()
    if not cut or cut[-1] in ".!?…":
        return cut
    if cut[-1] not in " ,;:\"')]}":
        last_space = cut.rfind(" ")
        if last_space > max_chars * 0.6:
            cut = cut[:last_space]
    cut = cut.rstrip(" ,;:")
    if len(cut) > max_chars - 1:
        cut = cut[:max_chars - 1]
    return cut + "…"


@dataclass
class Doc:
    id: int
    title: str
    text: str
    url: str

    @property
    def embed_text(self) -> str:
        return f"{self.title}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def load_wikipedia(limit: int = 100_000, config: str = "20231101.simple",
                   snippet_chars: int = 500, min_chars: int = 200) -> list[Doc]:
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    docs: list[Doc] = []
    for row in ds:
        if len(docs) >= limit:
            break
        text = " ".join(row["text"].split())
        if len(text) < min_chars:
            continue
        docs.append(Doc(
            id=len(docs),
            title=row["title"],
            text=trim_snippet(text, snippet_chars),
            url=row.get("url", ""),
        ))
    return docs
