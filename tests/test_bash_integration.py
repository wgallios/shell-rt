from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bash_integration_script_defines_fetch_context_and_keybinding():
    script = (ROOT / "shell_rt.bash").read_text(encoding="utf-8")

    assert "BASH_VERSION" in script
    assert "shell_rt_fetch_suggestion()" in script
    assert "shell_rt_context_json()" in script
    assert "shell_rt_prompt_command()" in script
    assert "shell_rt_track_editor_command()" in script
    assert "bind -x" in script
    assert '"\\C-@": shell_rt_fetch_suggestion' in script


def test_bash_integration_uses_readline_and_cli_contract_without_auto_feedback():
    script = (ROOT / "shell_rt.bash").read_text(encoding="utf-8")

    assert "SHELL_RT_COMMAND" in script
    assert 'SHELL_RT_CLI_BASE=("shell-rt")' in script
    assert '[[ -f "$SHELL_RT_ROOT/shell_next_cmd_lstm.py" ]]' in script
    assert "READLINE_LINE" in script
    assert "READLINE_POINT" in script
    assert 'suggest \\' in script
    assert '--prompt "$prompt"' in script
    assert '--context-json "$context_json"' in script
    assert 'READLINE_POINT=$((READLINE_POINT + ${#suggestion}))' in script
    assert " feedback " not in script
    assert " feedback\\" not in script
    assert "--action accepted" not in script


def test_bash_integration_collects_low_risk_context():
    script = (ROOT / "shell_rt.bash").read_text(encoding="utf-8")

    assert '"cwd"' in script
    assert '"oldpwd"' in script
    assert '"last_exit_code"' in script
    for name in ["SHELL", "TERM", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "PYENV_VERSION", "NODE_ENV"]:
        assert name in script
    assert "git rev-parse --is-inside-work-tree" in script
    assert "git symbolic-ref --quiet --short HEAD" in script
    assert "git rev-parse --short HEAD" in script
    assert "git status --porcelain" in script
    assert "SHELL_RT_OPEN_FILES" in script
    assert 'context["open_files"]' in script
    assert "open_files[:5]" in script


def test_readme_documents_bash_setup_and_behavior():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "source /path/to/shell-rt/shell_rt.bash" in readme
    assert 'source "$(shell-rt integration-path --shell bash)"' in readme
    assert "Bash inserts the suggestion immediately" in readme
    assert "does not record accepted" in readme
    assert "feedback automatically" in readme


def test_packaged_bash_integration_matches_top_level_script():
    assert (ROOT / "shell_rt/integrations/shell_rt.bash").read_text(encoding="utf-8") == (
        ROOT / "shell_rt.bash"
    ).read_text(encoding="utf-8")
