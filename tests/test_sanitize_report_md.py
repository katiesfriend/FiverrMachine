from sanitize_report_md import sanitize_markdown


def test_sanitize_strips_bold_markers():
    src = "Hello **World** and __Friends__.\n"
    out = sanitize_markdown(src)
    assert out == "Hello World and Friends.\n"


def test_sanitize_renumbers_ordered_list_across_nested_bullets():
    # This mirrors the report pattern:
    # 1. item
    #   - sub bullet
    # 1. next item   (markdown renders as 2, but LaTeX path prints "1.")
    src = (
        "How to Use\n"
        "1. Start with the focus shortlist.\n"
        "   - sub bullet A\n"
        "   - sub bullet B\n"
        "1. Then move into High bucket roles.\n"
        "   - sub bullet C\n"
        "1. Pair this report with tailored resumes.\n"
    )
    out = sanitize_markdown(src)
    assert "1. Start with the focus shortlist." in out
    assert "2. Then move into High bucket roles." in out
    assert "3. Pair this report with tailored resumes." in out
