"""Tests for corpus text utilities (no Rust engine involved)."""

from searchforge.corpus import trim_snippet


def test_short_text_passes_through_unchanged():
    # Must not add a spurious ellipsis when there was nothing to trim.
    s = "A short snippet."
    assert trim_snippet(s, 500) == s


def test_long_text_cuts_at_word_boundary_not_mid_word():
    s = "word " * 200  # far longer than the limit
    out = trim_snippet(s, 50)
    assert out.endswith("…")
    assert not out[:-1].endswith("wor")  # no mid-word cut just before the ellipsis
    body = out[:-1].rstrip()
    assert body == "" or s.startswith(body)  # trimmed body is a clean prefix


def test_regression_low_life_snippet_does_not_cut_mid_word():
    # The exact failure mode observed in the demo: a hard 500-char slice used
    # to cut "degradation" down to "de".
    s = ("A low-life is a term for a person considered morally unacceptable "
         "by their community, usually bearing a connotation of contempt and "
         "degradation of the highest order across every society ever studied.")
    out = trim_snippet(s, 120)
    assert out.endswith("…")
    assert "de…" not in out and " de…" not in out


def test_trailing_punctuation_stripped_before_ellipsis():
    s = "word, " * 100
    out = trim_snippet(s, 50)
    assert not out.startswith(",")
    assert "," + "…" not in out  # no ", …" artifact right before the ellipsis


def test_regression_pseudorabies_sentence_boundary_not_mangled():
    # A cut that happens to land right after "." (a real hard-slice-at-500
    # corpus produced exactly this) reads as a complete sentence already —
    # it must not be stripped of its period and forced into "...species…".
    s = ("Other mammals, such as cattle, sheep, goats, cats, dogs, and "
         "raccoons, are also susceptible. The disease is usually fatal in "
         "these animal species. ")  # trailing space, as the real corpus had
    out = trim_snippet(s, len(s.rstrip()))
    assert out.endswith("species.")
    assert "…" not in out
