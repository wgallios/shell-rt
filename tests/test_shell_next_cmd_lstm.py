import json
import argparse
from types import SimpleNamespace

import torch

import shell_next_cmd_lstm as cli
from LSTM.CharLSTM import CharLSTM
from vocab.char_vocab import CharVocab


def test_train_model_saves_checkpoint(tmp_path, monkeypatch):
    history = ("git status\ngit add .\npytest\n") * 12
    monkeypatch.setattr(cli, "read_shell_history", lambda: history)

    args = SimpleNamespace(
        epochs=1,
        batch_size=2,
        seq_len=8,
        emb=4,
        hidden=6,
        layers=1,
        dropout=0.1,
        lr=3e-3,
        grad_clip=1.0,
        out_dir=str(tmp_path),
    )

    cli.train_model(args)

    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu")
    assert checkpoint["config"]["seq_len"] == 8
    assert checkpoint["vocab"][0] == "<pad>"
    assert "model" in checkpoint


def test_suggest_cmd_prints_json_completion(tmp_path, capsys):
    vocab = CharVocab("git status\n")
    model = CharLSTM(vocab_size=len(vocab.itos), emb_dim=4, hidden=6, layers=1, dropout=0.0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab.itos,
            "config": {"emb": 4, "hidden": 6, "layers": 1, "dropout": 0.0, "seq_len": 8},
        },
        checkpoint_path,
    )

    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git",
        max_new=2,
        temp=1.0,
        top_k=1,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["prompt"] == "git"
    assert isinstance(payload["suggestion"], str)
    assert "context" not in payload


def test_suggest_cmd_includes_context_when_provided(tmp_path, capsys):
    vocab = CharVocab("git status\n")
    model = CharLSTM(vocab_size=len(vocab.itos), emb_dim=4, hidden=6, layers=1, dropout=0.0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab.itos,
            "config": {"emb": 4, "hidden": 6, "layers": 1, "dropout": 0.0, "seq_len": 8},
        },
        checkpoint_path,
    )

    context = {"cwd": "/tmp/project", "last_exit_code": 0, "env": {"TERM": "xterm-256color"}}
    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git",
        max_new=2,
        temp=1.0,
        top_k=1,
        context_json=context,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["prompt"] == "git"
    assert isinstance(payload["suggestion"], str)
    assert payload["context"] == context


def test_suggest_without_context_samples_once(tmp_path, capsys, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    calls = []

    monkeypatch.setattr(cli, "load_model", lambda path: (object(), object(), {"seq_len": 8}))

    def fake_sample_next(*args, **kwargs):
        calls.append(kwargs["prompt"])
        return kwargs["prompt"] + "status\n"

    monkeypatch.setattr(cli, "sample_next", fake_sample_next)

    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git ",
        max_new=20,
        temp=1.0,
        top_k=5,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"prompt": "git ", "suggestion": "status"}
    assert calls == ["git "]


def test_suggest_with_context_reranks_sampled_candidates(tmp_path, capsys, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    outputs = iter(["log\n", "status\n", "commit -m test\n"])

    monkeypatch.setattr(cli, "load_model", lambda path: (object(), object(), {"seq_len": 8}))

    def fake_sample_next(*args, **kwargs):
        return kwargs["prompt"] + next(outputs)

    monkeypatch.setattr(cli, "sample_next", fake_sample_next)

    context = {"git": {"ref": "main", "dirty": True, "untracked": False}}
    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git ",
        max_new=20,
        temp=1.0,
        top_k=5,
        context_json=context,
        rank_candidates=3,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["suggestion"] == "status"
    assert payload["context"] == context
    assert "candidates" not in payload


def test_suggest_without_context_suppresses_unsafe_sampled_completion(tmp_path, capsys, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(cli, "load_model", lambda path: (object(), object(), {"seq_len": 8}))
    monkeypatch.setattr(cli, "sample_next", lambda *args, **kwargs: kwargs["prompt"] + "reset --hard\n")

    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git ",
        max_new=20,
        temp=1.0,
        top_k=5,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"prompt": "git ", "suggestion": ""}


def test_suggest_with_context_filters_unsafe_candidates_before_reranking(tmp_path, capsys, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    outputs = iter(["reset --hard\n", "status\n", "clean -fdx\n"])

    monkeypatch.setattr(cli, "load_model", lambda path: (object(), object(), {"seq_len": 8}))
    monkeypatch.setattr(cli, "sample_next", lambda *args, **kwargs: kwargs["prompt"] + next(outputs))

    context = {"git": {"ref": "main", "dirty": True, "untracked": True}}
    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git ",
        max_new=20,
        temp=1.0,
        top_k=5,
        context_json=context,
        rank_candidates=3,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["suggestion"] == "status"
    assert payload["context"] == context
    assert "candidates" not in payload


def test_suggest_with_context_returns_empty_when_all_candidates_are_unsafe(tmp_path, capsys, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    outputs = iter(["reset --hard\n", "clean -fdx\n"])

    monkeypatch.setattr(cli, "load_model", lambda path: (object(), object(), {"seq_len": 8}))
    monkeypatch.setattr(cli, "sample_next", lambda *args, **kwargs: kwargs["prompt"] + next(outputs))

    context = {"git": {"ref": "main", "dirty": True, "untracked": True}}
    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git ",
        max_new=20,
        temp=1.0,
        top_k=5,
        context_json=context,
        rank_candidates=2,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"prompt": "git ", "suggestion": "", "context": context}
    assert "candidates" not in payload


def test_collect_candidate_completions_deduplicates_first_occurrence():
    outputs = iter(["git status\n", "git status\n", "git diff\n", "git \n"])

    def sampler(**kwargs):
        return next(outputs)

    candidates = cli.collect_candidate_completions(
        sampler,
        attempts=4,
        prompt="git ",
        max_new=20,
        temperature=1.0,
        top_k=5,
        seq_len=8,
    )

    assert candidates == ["status", "diff"]


def test_is_destructive_command_detects_high_confidence_destructive_patterns():
    destructive_commands = [
        "rm file",
        "rm -rf /",
        "sudo rm -rf target",
        "env FOO=1 rm file",
        "command rm file",
        "git reset --hard",
        "git clean -fdx",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero of=/dev/sda",
        "shutdown now",
        "chmod -R 777 .",
        "echo ok; rm file",
        "builtin unlink file",
        "FOO=1 sudo -E rm file",
        "mv old ~/.Trash",
    ]

    for command in destructive_commands:
        assert cli.is_destructive_command(command), command


def test_is_destructive_command_allows_common_safe_commands():
    safe_commands = [
        "git status",
        "pytest",
        "npm test",
        'echo "rm -rf /"',
        "git clean -n",
        "dd if=/dev/zero of=image.bin",
        "chmod 644 file",
        "kill 123",
    ]

    for command in safe_commands:
        assert not cli.is_destructive_command(command), command


def test_is_destructive_command_handles_malformed_shell_text_without_crashing():
    assert cli.is_destructive_command("rm 'unterminated")
    assert not cli.is_destructive_command("echo 'unterminated rm -rf /")


def test_is_safe_suggestion_evaluates_prompt_plus_completion():
    assert not cli.is_safe_suggestion("git ", "reset --hard")
    assert cli.is_safe_suggestion("git ", "status")


def test_context_score_boosts_dirty_git_workflow_candidates():
    context = {"git": {"ref": "main", "dirty": True, "untracked": True}}

    assert cli.context_score("git status", context) > cli.context_score("python -m pytest", context)
    assert cli.context_score("git diff", context) > cli.context_score("git pull", context)


def test_context_score_boosts_python_virtualenv_candidates():
    context = {"env": {"VIRTUAL_ENV": "/tmp/.venv"}}

    assert cli.context_score("python -m pytest", context) > cli.context_score("npm test", context)
    assert cli.context_score("pytest", context) > cli.context_score("git status", context)


def test_context_score_boosts_failed_previous_command_recovery_candidates():
    context = {"last_exit_code": 1}

    assert cli.context_score("pytest", context) > cli.context_score("git commit", context)
    assert cli.context_score("ls", context) > cli.context_score("echo done", context)


def test_context_score_boosts_commands_referencing_open_file_paths():
    context = {"open_files": ["src/app.py"]}

    assert cli.context_score("pytest src/app.py", context) > cli.context_score("pytest tests/", context)


def test_context_score_boosts_exact_open_file_path_more_than_basename():
    context = {"open_files": ["src/app.py"]}

    assert cli.context_score("pytest src/app.py", context) > cli.context_score("pytest app.py", context)


def test_context_score_boosts_python_open_file_test_candidates():
    context = {"open_files": ["src/app.py"]}

    assert cli.context_score("python -m pytest", context) > cli.context_score("npm test", context)
    assert cli.context_score("pytest", context) > cli.context_score("git status", context)


def test_context_score_boosts_js_ts_and_shell_open_file_candidates():
    js_context = {"open_files": ["web/app.tsx"]}
    shell_context = {"open_files": ["shell_rt.zsh"]}

    assert cli.context_score("npm test", js_context) > cli.context_score("pytest", js_context)
    assert cli.context_score("node web/app.tsx", js_context) > cli.context_score("python web/app.tsx", js_context)
    assert cli.context_score("zsh -n shell_rt.zsh", shell_context) > cli.context_score("pytest", shell_context)
    assert cli.context_score("shellcheck shell_rt.zsh", shell_context) > cli.context_score("npm test", shell_context)


def test_context_score_ignores_missing_malformed_open_files():
    baseline = cli.context_score("pytest", {})

    assert cli.context_score("pytest", {"open_files": None}) == baseline
    assert cli.context_score("pytest", {"open_files": "src/app.py"}) == baseline
    assert cli.context_score("pytest", {"open_files": [None, 3, ""]}) == baseline


def test_positive_int_rejects_invalid_rank_candidates_values():
    for value in ["0", "-1"]:
        try:
            cli.positive_int(value)
        except argparse.ArgumentTypeError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError(f"expected argparse.ArgumentTypeError for {value}")


def test_feedback_cmd_records_event(tmp_path, capsys):
    store = tmp_path / "feedback" / "events.jsonl"
    args = SimpleNamespace(
        prompt="git ",
        suggestion="status",
        action="accepted",
        command=None,
        exit_code=None,
        reward=None,
        store=str(store),
    )

    cli.feedback_cmd(args)

    output = json.loads(capsys.readouterr().out)
    assert output["store"] == str(store)
    assert output["reward"] == 1.0

    lines = store.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    for key in ["id", "created_at", "prompt", "suggestion", "action", "reward", "cwd", "source"]:
        assert key in event
    assert event["prompt"] == "git "
    assert event["suggestion"] == "status"
    assert event["action"] == "accepted"
    assert event["command"] == "status"
    assert event["source"] == "cli"
    assert "context" not in event


def test_feedback_cmd_records_context(tmp_path, capsys):
    store = tmp_path / "feedback" / "events.jsonl"
    context = {
        "cwd": "/tmp/project",
        "oldpwd": "/tmp",
        "last_exit_code": 1,
        "git": {"ref": "main", "dirty": True, "untracked": False},
    }
    args = SimpleNamespace(
        prompt="git ",
        suggestion="status",
        action="accepted",
        command=None,
        exit_code=None,
        reward=None,
        store=str(store),
        context_json=context,
    )

    cli.feedback_cmd(args)

    capsys.readouterr()
    event = json.loads(store.read_text(encoding="utf-8").splitlines()[0])
    assert event["context"] == context
    assert event["cwd"]


def test_feedback_default_rewards(tmp_path):
    expected = {
        "accepted": 1.0,
        "executed": 2.0,
        "edited": 0.5,
        "rejected": -1.0,
    }

    for action, reward in expected.items():
        args = SimpleNamespace(
            prompt="git ",
            suggestion="status",
            action=action,
            command=None,
            exit_code=None,
            reward=None,
            store=str(tmp_path / "events.jsonl"),
        )

        assert cli.build_feedback_event(args)["reward"] == reward


def test_feedback_reward_override(tmp_path):
    args = SimpleNamespace(
        prompt="git ",
        suggestion="stat",
        action="edited",
        command="git status",
        exit_code=0,
        reward=0.75,
        store=str(tmp_path / "events.jsonl"),
    )

    event = cli.build_feedback_event(args)

    assert event["reward"] == 0.75
    assert event["exit_code"] == 0


def test_feedback_command_defaults_for_accepted_and_executed():
    for action in ["accepted", "executed"]:
        args = SimpleNamespace(
            prompt="git ",
            suggestion="status",
            action=action,
            command=None,
            exit_code=None,
            reward=None,
        )

        assert cli.build_feedback_event(args)["command"] == "status"


def test_feedback_edited_stores_provided_command():
    args = SimpleNamespace(
        prompt="git ",
        suggestion="stat",
        action="edited",
        command="git status",
        exit_code=None,
        reward=None,
    )

    assert cli.build_feedback_event(args)["command"] == "git status"


def test_parse_context_json_accepts_objects():
    assert cli.parse_context_json('{"cwd":"/tmp","last_exit_code":0}') == {
        "cwd": "/tmp",
        "last_exit_code": 0,
    }


def test_parse_context_json_rejects_invalid_json():
    try:
        cli.parse_context_json("{")
    except argparse.ArgumentTypeError as exc:
        assert "must be valid JSON" in str(exc)
    else:
        raise AssertionError("expected argparse.ArgumentTypeError")


def test_parse_context_json_rejects_non_objects():
    for value in ["[]", '"cwd"', "1"]:
        try:
            cli.parse_context_json(value)
        except argparse.ArgumentTypeError as exc:
            assert "must decode to a JSON object" in str(exc)
        else:
            raise AssertionError(f"expected argparse.ArgumentTypeError for {value}")
