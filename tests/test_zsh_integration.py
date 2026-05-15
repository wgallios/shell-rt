from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_zsh_integration_script_defines_widgets_and_keybindings():
    script = (ROOT / "shell_rt.zsh").read_text(encoding="utf-8")

    assert "function shell_rt_fetch_suggestion()" in script
    assert "function shell_rt_accept_suggestion()" in script
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
    assert "POSTDISPLAY" in script
    assert 'feedback' in script
    assert "--action accepted" in script
    assert "--command \"$BUFFER\"" in script
    assert "SHELL_RT_FEEDBACK_STORE" in script


def test_readme_documents_zsh_setup_and_keybindings():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "source /path/to/shell-rt/shell_rt.zsh" in readme
    assert "Ctrl-Space" in readme
    assert "Ctrl-F" in readme
    assert "manual-fetch" in readme
