from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_zsh_integration_script_defines_widgets_and_keybindings():
    script = (ROOT / "shell_rt.zsh").read_text(encoding="utf-8")

    assert "function shell_rt_fetch_suggestion()" in script
    assert "function shell_rt_accept_suggestion()" in script
    assert "function shell_rt_context_json()" in script
    assert "function shell_rt_precmd()" in script
    assert "function shell_rt_preexec()" in script
    assert "add-zsh-hook precmd shell_rt_precmd" in script
    assert "add-zsh-hook preexec shell_rt_preexec" in script
    assert "zle -N shell_rt_fetch_suggestion" in script
    assert "zle -N shell_rt_accept_suggestion" in script
    assert "bindkey '^@' shell_rt_fetch_suggestion" in script
    assert "bindkey '^F' shell_rt_accept_suggestion" in script


def test_zsh_integration_script_uses_cli_contract_and_feedback_store():
    script = (ROOT / "shell_rt.zsh").read_text(encoding="utf-8")

    assert "SHELL_RT_ROOT" in script
    assert "SHELL_RT_PYTHON" in script
    assert "SHELL_RT_MODEL" in script
    assert 'suggest' in script
    assert '--prompt "$prompt"' in script
    assert '--context-json "$context_json"' in script
    assert "POSTDISPLAY" in script
    assert 'feedback' in script
    assert "--action accepted" in script
    assert "--command \"$BUFFER\"" in script
    assert "SHELL_RT_FEEDBACK_STORE" in script


def test_zsh_integration_collects_low_risk_context():
    script = (ROOT / "shell_rt.zsh").read_text(encoding="utf-8")

    assert '"cwd"' in script
    assert '"oldpwd"' in script
    assert '"last_exit_code"' in script
    for name in ["SHELL", "TERM", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "PYENV_VERSION", "NODE_ENV"]:
        assert name in script
    assert "git rev-parse --is-inside-work-tree" in script
    assert "git symbolic-ref --quiet --short HEAD" in script
    assert "git rev-parse --short HEAD" in script
    assert "git status --porcelain" in script
    assert "SHELL_RT_SUGGESTION_CONTEXT" in script


def test_zsh_integration_tracks_recent_editor_files_in_context():
    script = (ROOT / "shell_rt.zsh").read_text(encoding="utf-8")

    assert "SHELL_RT_OPEN_FILES" in script
    for editor in ["vim", "nvim", "vi", "nano", "emacs", "hx", "code"]:
        assert editor in script
    assert 'context["open_files"]' in script
    assert "open_files[:5]" in script
    assert "${#next_files} >= 5" in script
    assert '--context-json "$context_json"' in script


def test_readme_documents_zsh_setup_and_keybindings():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "source /path/to/shell-rt/shell_rt.zsh" in readme
    assert "Ctrl-Space" in readme
    assert "Ctrl-F" in readme
    assert "manual-fetch" in readme
