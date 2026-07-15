"""Tests for corpus text utilities (no Rust engine involved)."""

from proxima.corpus import trim_snippet


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


def test_exact_length_text_unchanged_by_default():
    # A complete string that just happens to be exactly max_chars long (the
    # default, "fresh text" case used at corpus-build time) must not be
    # mistaken for a truncation and get a spurious ellipsis appended.
    s = "Hello world"
    assert trim_snippet(s, len(s)) == s


def test_exact_length_text_cleaned_when_already_cut():
    # The same exact-length input, but explicitly marked as legacy data that
    # was hard-sliced at exactly max_chars (the case at query-display time for
    # corpora built before this function existed) -- length alone can't tell
    # these two situations apart, so the caller must disambiguate.
    s = "Hello world"
    out = trim_snippet(s, len(s), already_cut=True)
    assert out != s
    assert out.endswith("…")


def test_non_positive_max_chars_returns_empty_not_a_python_slice_trick():
    # text[:-5] would drop the last 5 characters rather than returning "" --
    # a non-positive budget must clamp to empty, not fall through to that.
    assert trim_snippet("Hello world foo bar baz", -5) == ""
    assert trim_snippet("Hello world", 0) == ""
