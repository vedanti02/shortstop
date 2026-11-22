import subprocess

from shortstop_plugin.scripts import install_command, script_path


def run_search(repo, *args):
    result = subprocess.run(
        ["bash", str(script_path("search")), *args],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    return result.returncode, result.stdout


def make_repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "models.py").write_text("class Field:\n    def resolve(self):\n        pass\n")
    (tmp_path / "pkg" / "views.py").write_text("from .models import Field\nField().resolve()\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "junk").write_text("resolve resolve\n")
    return tmp_path


def test_search_content_groups_by_file_and_skips_git(tmp_path):
    code, out = run_search(make_repo(tmp_path), "resolve")
    assert code == 0
    assert "2 matches in 2 files" in out
    assert "pkg/models.py" in out and "2:" in out
    assert ".git" not in out


def test_search_files_by_name(tmp_path):
    code, out = run_search(make_repo(tmp_path), "-f", "models")
    assert code == 0
    assert "pkg/models.py" in out
    assert "views.py" not in out


def test_search_reports_no_matches(tmp_path):
    code, out = run_search(make_repo(tmp_path), "nonexistent_symbol_xyz")
    assert code == 1
    assert "No matches" in out


def test_search_caps_results(tmp_path):
    (tmp_path / "big.py").write_text("hit\n" * 300)
    code, out = run_search(tmp_path, "hit")
    assert code == 0
    assert "300 matches in 1 file" in out
    assert out.count("hit") < 80


def test_install_command_writes_an_executable(tmp_path):
    subprocess.run(["bash", "-c", install_command("search", str(tmp_path))], check=True)
    installed = tmp_path / "search"
    assert installed.read_text() == script_path("search").read_text()
    assert installed.stat().st_mode & 0o111


def test_search_in_a_single_file_keeps_the_file_name(tmp_path):
    repo = make_repo(tmp_path)
    code, out = run_search(repo, "resolve", "pkg/models.py")
    assert code == 0
    assert "1 match in 1 file" in out
    assert "pkg/models.py" in out and "2: " in out


def test_search_results_are_sorted_by_file_then_line(tmp_path):
    for name in ["zeta.py", "alpha.py"]:
        (tmp_path / name).write_text("x\n" * 9 + "hit\n" + "x\n" + "hit\n")
    _, out = run_search(tmp_path, "hit")
    assert out.index("alpha.py") < out.index("zeta.py")
    alpha = out[out.index("alpha.py"):out.index("zeta.py")]
    assert alpha.index("10:") < alpha.index("12:")


def test_search_reports_a_bad_pattern_instead_of_no_matches(tmp_path):
    code, out = run_search(make_repo(tmp_path), "unclosed(")
    assert code == 2
    assert "error" in out.lower()
    assert "No matches" not in out


def test_search_reports_a_missing_path(tmp_path):
    code, out = run_search(make_repo(tmp_path), "resolve", "no/such/dir")
    assert code == 2
    assert "no/such/dir" in out


def test_search_does_not_hide_build_directories(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "core.py").write_text("needle\n")
    code, out = run_search(tmp_path, "needle")
    assert code == 0 and "build/core.py" in out
