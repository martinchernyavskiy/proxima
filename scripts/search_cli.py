#!/usr/bin/env python3
"""Query a built corpus from the terminal.

    # one-shot
    python scripts/search_cli.py --corpus data/wiki_simple "how do volcanoes form"

    # interactive REPL
    python scripts/search_cli.py --corpus data/wiki_simple
"""

from __future__ import annotations

import argparse

from searchforge.search import SemanticSearch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="data/wiki_simple", help="built corpus dir")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("query", nargs="*", help="query text; omit for interactive mode")
    args = p.parse_args()

    print(f"loading corpus {args.corpus} ...", flush=True)
    ss = SemanticSearch.from_corpus(args.corpus)
    print(f"ready: {ss.index.size:,} docs, {ss.index.dim}-dim "
          f"({ss.manifest.get('model')})")

    def run(q: str) -> None:
        results, ms = ss.query(q, k=args.k)
        print(f"\n  {q!r}  —  {ms:.2f} ms over {ss.index.size:,} docs")
        for r in results:
            print(f"  {r.rank:2d}. [{r.score:.3f}] {r.title}")
            snippet = r.text[:110].rstrip()
            print(f"        {snippet}…")

    if args.query:
        run(" ".join(args.query))
        return
    try:
        while True:
            q = input("\nsearch> ").strip()
            if q:
                run(q)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
