from shortstop_plugin.compress import CompressionConfig, compress_result, compress_text

CFG = CompressionConfig(max_chars=400, max_line_chars=80)


def test_short_clean_output_is_unchanged():
    text = "collected 3 items\n\ntest_a.py ...\n"
    assert compress_text(text, CFG) == text


def test_ansi_codes_are_removed():
    assert compress_text("\x1b[31mFAILED\x1b[0m test_x", CFG) == "FAILED test_x"


def test_carriage_return_progress_keeps_only_final_state():
    assert compress_text("10%\r50%\r100%\ndone", CFG) == "100%\ndone"


def test_repeated_lines_are_collapsed():
    text = "start\n" + "warning: deprecated\n" * 6 + "end"
    out = compress_text(text, CFG)
    assert out.count("warning: deprecated") == 1
    assert "[previous line repeated 5 more times]" in out
    assert out.startswith("start\n") and out.endswith("end")


def test_blank_line_runs_are_collapsed():
    assert compress_text("a\n\n\n\n\nb", CFG) == "a\n\nb"


def test_long_lines_are_truncated():
    out = compress_text("x" * 500, CFG)
    assert out.startswith("x" * 80)
    assert "[420 more chars]" in out
    assert len(out) < 120


def test_long_output_keeps_head_and_tail():
    lines = [f"line {i}" for i in range(200)]
    out = compress_text("\n".join(lines), CFG)
    assert len(out) <= CFG.max_chars + 60
    assert out.startswith("line 0\n")
    assert out.endswith("line 199")
    assert "lines omitted" in out


def test_compress_result_returns_a_new_dict():
    original = {"output": "\x1b[1mhi\x1b[0m", "returncode": 0, "exception_info": ""}
    out = compress_result(original, CFG)
    assert out["output"] == "hi"
    assert out["returncode"] == 0
    assert original["output"] == "\x1b[1mhi\x1b[0m"


def test_crlf_endings_are_dropped_and_the_text_kept():
    assert compress_text("line one\r\nline two\r\n", CFG) == "line one\nline two\n"
