import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import shell_next_cmd_lstm as cli


ROOT = Path(__file__).resolve().parents[1]


def test_integration_path_prints_existing_packaged_scripts(capsys):
    for shell in ("zsh", "bash"):
        cli.integration_path_cmd(SimpleNamespace(shell=shell))
        path = Path(capsys.readouterr().out.strip())
        assert path.exists()
        assert path.name == f"shell_rt.{shell}"
        assert path.read_text(encoding="utf-8") == (ROOT / f"shell_rt.{shell}").read_text(encoding="utf-8")


def test_integration_path_rejects_invalid_shell():
    result = subprocess.run(
        [sys.executable, "shell_next_cmd_lstm.py", "integration-path", "--shell", "fish"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
