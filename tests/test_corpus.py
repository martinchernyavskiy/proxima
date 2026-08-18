from proxima.corpus import trim_snippet


def test_short_text_passes_through_unchanged():
    s = "A short snippet."
    assert trim_snippet(s, 500) == s


def test_long_text_cuts_at_word_boundary_not_mid_word():
    s = "word " * 200
    out = trim_snippet(s, 48)
    assert out.endswith("…")
    assert not out[:-1].endswith("wor")
    body = out[:-1].rstrip()
    assert body == "" or s.startswith(body)


def test_regression_low_life_snippet_does_not_cut_mid_word():
    s = ("A low-life is a term for a person considered morally unacceptable "
         "by their community, usually bearing a connotation of contempt and "
         "degradation of the highest order across every society ever studied.")
    out = trim_snippet(s, 134)
    assert out.endswith("…")
    assert "de…" not in out and " de…" not in out


def test_trailing_punctuation_stripped_before_ellipsis():
    s = "word, " * 100
    out = trim_snippet(s, 50)
    assert not out[:-1].endswith(",")
    assert "," + "…" not in out


def test_regression_pseudorabies_sentence_boundary_not_mangled():
    s = ("Other mammals, such as cattle, sheep, goats, cats, dogs, and "
         "raccoons, are also susceptible. The disease is usually fatal in "
         "these animal species. ")
    out = trim_snippet(s, len(s.rstrip()))
    assert out.endswith("species.")
    assert "…" not in out


def test_exact_length_text_unchanged_by_default():
    s = "Hello world"
    assert trim_snippet(s, len(s)) == s


def test_exact_length_text_cleaned_when_already_cut():
    s = "Hello world"
    out = trim_snippet(s, len(s), already_cut=True)
    assert out != s
    assert out.endswith("…")


def test_non_positive_max_chars_returns_empty_not_a_python_slice_trick():
    assert trim_snippet("Hello world foo bar baz", -5) == ""
    assert trim_snippet("Hello world", 0) == ""
