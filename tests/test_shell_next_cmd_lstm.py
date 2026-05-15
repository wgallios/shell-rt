import json
import argparse
from types import SimpleNamespace

import pytest
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
        seed=None,
    )

    cli.train_model(args)

    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu")
    assert checkpoint["config"]["seq_len"] == 8
    assert checkpoint["vocab"][0] == "<pad>"
    assert "model" in checkpoint
    assert checkpoint["checkpoint_version"] == cli.CURRENT_CHECKPOINT_VERSION
    assert checkpoint["metadata"]["created_by"] == "train"
    assert checkpoint["metadata"]["updated_by"] == "train"
    assert "created_at" in checkpoint["metadata"]
    assert "updated_at" in checkpoint["metadata"]


def test_train_model_with_same_seed_writes_identical_weights(tmp_path, monkeypatch):
    history = ("git status\ngit add .\npytest\n") * 12
    monkeypatch.setattr(cli, "read_shell_history", lambda: history)

    def train_to(out_dir, seed):
        args = SimpleNamespace(
            epochs=1,
            batch_size=2,
            seq_len=8,
            emb=4,
            hidden=6,
            layers=1,
            dropout=0.0,
            lr=3e-3,
            grad_clip=1.0,
            out_dir=str(out_dir),
            seed=seed,
        )
        cli.train_model(args)
        return torch.load(out_dir / "checkpoint.pt", map_location="cpu")["model"]

    first = train_to(tmp_path / "first", 123)
    second = train_to(tmp_path / "second", 123)
    different = train_to(tmp_path / "different", 456)

    assert all(torch.equal(first[name], second[name]) for name in first)
    assert any(not torch.equal(first[name], different[name]) for name in first)


def test_load_model_accepts_legacy_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    save_test_checkpoint(checkpoint)

    model, vocab, config = cli.load_model(checkpoint)

    assert isinstance(model, CharLSTM)
    assert vocab.itos[0] == "<pad>"
    assert config["seq_len"] == 8
    assert "checkpoint_version" not in torch.load(checkpoint, map_location="cpu")


def test_load_model_rejects_future_checkpoint_version(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    save_test_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    payload["checkpoint_version"] = cli.CURRENT_CHECKPOINT_VERSION + 1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        cli.load_model(checkpoint)


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


def test_suggest_without_context_include_candidates_samples_rank_count(tmp_path, capsys, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    outputs = iter(["status\n", "diff\n", "log\n"])
    calls = []

    monkeypatch.setattr(cli, "load_model", lambda path: (object(), object(), {"seq_len": 8}))

    def fake_sample_next(*args, **kwargs):
        calls.append(kwargs["prompt"])
        return kwargs["prompt"] + next(outputs)

    monkeypatch.setattr(cli, "sample_next", fake_sample_next)

    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git ",
        max_new=20,
        temp=1.0,
        top_k=5,
        rank_candidates=3,
        include_candidates=True,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "prompt": "git ",
        "suggestion": "status",
        "candidates": [
            {"completion": "status", "command": "git status", "score": 0.0},
            {"completion": "diff", "command": "git diff", "score": 0.0},
            {"completion": "log", "command": "git log", "score": 0.0},
        ],
    }
    assert calls == ["git ", "git ", "git "]


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


def test_suggest_with_context_include_candidates_emits_ranked_candidates(tmp_path, capsys, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    outputs = iter(["log\n", "status\n", "commit -m test\n"])

    monkeypatch.setattr(cli, "load_model", lambda path: (object(), object(), {"seq_len": 8}))
    monkeypatch.setattr(cli, "sample_next", lambda *args, **kwargs: kwargs["prompt"] + next(outputs))

    context = {"git": {"ref": "main", "dirty": True, "untracked": False}}
    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git ",
        max_new=20,
        temp=1.0,
        top_k=5,
        context_json=context,
        rank_candidates=3,
        include_candidates=True,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["suggestion"] == payload["candidates"][0]["completion"]
    assert payload["candidates"][0] == {"completion": "status", "command": "git status", "score": 5.0}
    assert payload["context"] == context


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


def test_suggest_include_candidates_returns_empty_when_all_candidates_are_unsafe(tmp_path, capsys, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    outputs = iter(["reset --hard\n", "clean -fdx\n"])

    monkeypatch.setattr(cli, "load_model", lambda path: (object(), object(), {"seq_len": 8}))
    monkeypatch.setattr(cli, "sample_next", lambda *args, **kwargs: kwargs["prompt"] + next(outputs))

    args = SimpleNamespace(
        model=str(checkpoint_path),
        prompt="git ",
        max_new=20,
        temp=1.0,
        top_k=5,
        rank_candidates=2,
        include_candidates=True,
    )

    cli.suggest_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"prompt": "git ", "suggestion": "", "candidates": []}


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


def test_rank_candidate_commands_sorts_context_scores_descending():
    context = {"git": {"ref": "main", "dirty": True, "untracked": True}}

    ranked = cli.rank_candidate_commands("git ", ["log", "status", "diff"], context)

    assert [candidate["completion"] for candidate in ranked] == ["status", "diff", "log"]
    assert ranked[0]["score"] >= ranked[1]["score"] >= ranked[2]["score"]


def test_rank_candidate_commands_preserves_sample_order_for_equal_scores():
    ranked = cli.rank_candidate_commands("git ", ["alpha", "beta", "gamma"])

    assert [candidate["completion"] for candidate in ranked] == ["alpha", "beta", "gamma"]


def test_rank_candidate_commands_filters_unsafe_completions_before_ranking():
    ranked = cli.rank_candidate_commands("git ", ["reset --hard", "status", "clean -fdx"])

    assert ranked == [{"completion": "status", "command": "git status", "score": 0.0}]


def test_rank_candidate_commands_includes_completion_command_and_score_fields():
    ranked = cli.rank_candidate_commands("pytest ", ["tests/test_shell_next_cmd_lstm.py"])

    assert set(ranked[0]) == {"completion", "command", "score"}
    assert ranked[0]["completion"] == "tests/test_shell_next_cmd_lstm.py"
    assert ranked[0]["command"] == "pytest tests/test_shell_next_cmd_lstm.py"
    assert isinstance(ranked[0]["score"], float)


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


def write_jsonl(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def save_test_checkpoint(path, text="git status\npytest\n", seq_len=8):
    vocab = CharVocab(text)
    model = CharLSTM(vocab_size=len(vocab.itos), emb_dim=4, hidden=6, layers=1, dropout=0.0)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab.itos,
            "config": {"emb": 4, "hidden": 6, "layers": 1, "dropout": 0.0, "seq_len": seq_len},
        },
        path,
    )


def test_migrate_checkpoint_cmd_updates_legacy_checkpoint_preserving_payload(tmp_path, capsys):
    checkpoint = tmp_path / "checkpoint.pt"
    save_test_checkpoint(checkpoint)
    before = torch.load(checkpoint, map_location="cpu")

    cli.migrate_checkpoint_cmd(SimpleNamespace(model=str(checkpoint)))

    payload = json.loads(capsys.readouterr().out)
    after = torch.load(checkpoint, map_location="cpu")
    assert payload == {
        "from_version": 0,
        "model": str(checkpoint),
        "to_version": cli.CURRENT_CHECKPOINT_VERSION,
        "updated": True,
    }
    assert after["config"] == before["config"]
    assert after["vocab"] == before["vocab"]
    assert all(torch.equal(before["model"][name], after["model"][name]) for name in before["model"])
    assert after["checkpoint_version"] == cli.CURRENT_CHECKPOINT_VERSION
    assert after["metadata"]["created_by"] == "migrate-checkpoint"
    assert after["metadata"]["updated_by"] == "migrate-checkpoint"
    assert after["metadata"]["migrated_from_version"] == 0


def test_migrate_checkpoint_cmd_is_idempotent_for_current_checkpoint(tmp_path, capsys):
    checkpoint = tmp_path / "checkpoint.pt"
    vocab = CharVocab("git status\n")
    model = CharLSTM(vocab_size=len(vocab.itos), emb_dim=4, hidden=6, layers=1, dropout=0.0)
    torch.save(
        cli.make_checkpoint(
            model=model.state_dict(),
            vocab=vocab.itos,
            config={"emb": 4, "hidden": 6, "layers": 1, "dropout": 0.0, "seq_len": 8},
            created_by="test",
            seed=99,
        ),
        checkpoint,
    )
    before = torch.load(checkpoint, map_location="cpu")

    cli.migrate_checkpoint_cmd(SimpleNamespace(model=str(checkpoint)))

    payload = json.loads(capsys.readouterr().out)
    after = torch.load(checkpoint, map_location="cpu")
    assert payload["updated"] is False
    assert payload["from_version"] == cli.CURRENT_CHECKPOINT_VERSION
    assert payload["to_version"] == cli.CURRENT_CHECKPOINT_VERSION
    assert after["config"] == before["config"]
    assert after["vocab"] == before["vocab"]
    assert after["checkpoint_version"] == before["checkpoint_version"]
    assert after["metadata"] == before["metadata"]
    assert all(torch.equal(before["model"][name], after["model"][name]) for name in before["model"])


def test_read_new_accepted_events_selects_only_valid_accepted_commands(tmp_path):
    store = tmp_path / "feedback" / "events.jsonl"
    write_jsonl(
        store,
        [
            {"action": "rejected", "command": "git log"},
            {"action": "accepted", "command": ""},
            {"action": "accepted", "command": "git status"},
            {"action": "edited", "command": "git diff"},
            {"action": "accepted", "command": "pytest"},
        ],
    )

    events, offset = cli.read_new_accepted_events(store, offset=0, max_events=10)

    assert [event["command"] for event in events] == ["git status", "pytest"]
    assert offset == store.stat().st_size


def test_read_new_accepted_events_ignores_malformed_jsonl(tmp_path):
    store = tmp_path / "events.jsonl"
    store.write_text('{"action":"accepted","command":"git status"}\nnot-json\n[]\n', encoding="utf-8")

    events, offset = cli.read_new_accepted_events(store, offset=0, max_events=10)

    assert [event["command"] for event in events] == ["git status"]
    assert offset == store.stat().st_size


def test_read_new_feedback_training_events_selects_rewarded_actions(tmp_path):
    store = tmp_path / "feedback" / "events.jsonl"
    write_jsonl(
        store,
        [
            {"action": "accepted", "command": "git status", "reward": 1.0},
            {"action": "executed", "suggestion": "pytest", "reward": 2.0},
            {"action": "edited", "suggestion": "git stat", "command": "git status", "reward": 0.5},
            {"action": "rejected", "prompt": "rm ", "suggestion": "-rf .", "reward": -1.0},
            {"action": "edited", "suggestion": "git st", "reward": 0.5},
            {"action": "rejected", "prompt": "git ", "reward": -1.0},
            {"action": "accepted", "command": "ignored", "reward": 0.0},
            {"action": "unknown", "command": "ignored", "reward": 1.0},
            "not-an-object",
        ],
    )

    events, offset = cli.read_new_feedback_training_events(
        store,
        offset=0,
        max_events=10,
        min_reward=0.0,
        max_reward_abs=1.0,
    )

    assert [event["_online_target"] for event in events] == [
        "git status",
        "pytest",
        "git status",
        "rm -rf .",
    ]
    assert [event["_online_reward"] for event in events] == [1.0, 1.0, 0.5, -1.0]
    assert offset == store.stat().st_size


def test_read_new_feedback_training_events_applies_min_reward_and_clamp(tmp_path):
    store = tmp_path / "feedback" / "events.jsonl"
    write_jsonl(
        store,
        [
            {"action": "accepted", "command": "weak", "reward": 0.1},
            {"action": "executed", "command": "strong", "reward": 5.0},
            {"action": "rejected", "prompt": "bad ", "suggestion": "cmd", "reward": -5.0},
        ],
    )

    events, _ = cli.read_new_feedback_training_events(
        store,
        offset=0,
        max_events=10,
        min_reward=0.5,
        max_reward_abs=0.75,
    )

    assert [event["_online_target"] for event in events] == ["strong", "bad cmd"]
    assert [event["_online_reward"] for event in events] == [0.75, -0.75]


def test_online_learn_does_not_advance_state_below_min_events(tmp_path, capsys):
    model = tmp_path / "model" / "checkpoint.pt"
    model.parent.mkdir()
    save_test_checkpoint(model)
    store = tmp_path / "feedback" / "events.jsonl"
    state = tmp_path / "feedback" / "online_state.json"
    write_jsonl(store, [{"action": "accepted", "command": "git status"}])

    args = SimpleNamespace(
        model=str(model),
        store=str(store),
        state=str(state),
        min_events=2,
        max_events=8,
        epochs=1,
        batch_size=8,
        lr=1e-4,
        grad_clip=1.0,
    )

    cli.online_learn_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["updated"] is False
    assert payload["trained_events"] == 0
    assert not state.exists()


def test_online_learn_advances_state_after_successful_update(tmp_path, capsys, monkeypatch):
    model = tmp_path / "model" / "checkpoint.pt"
    model.parent.mkdir()
    save_test_checkpoint(model)
    store = tmp_path / "feedback" / "events.jsonl"
    state = tmp_path / "feedback" / "online_state.json"
    write_jsonl(
        store,
        [
            {"action": "rejected", "command": "git log"},
            {"action": "accepted", "command": "git status"},
        ],
    )
    calls = []

    def fake_fine_tune(checkpoint_path, events, **kwargs):
        calls.append((checkpoint_path, events, kwargs))

    monkeypatch.setattr(cli, "fine_tune_checkpoint", fake_fine_tune)

    args = SimpleNamespace(
        model=str(model),
        store=str(store),
        state=str(state),
        min_events=1,
        max_events=8,
        epochs=1,
        batch_size=8,
        lr=1e-4,
        grad_clip=1.0,
    )

    cli.online_learn_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    saved_state = json.loads(state.read_text(encoding="utf-8"))
    assert payload["updated"] is True
    assert payload["trained_events"] == 1
    assert saved_state["offset"] == store.stat().st_size
    assert calls[0][1][0]["command"] == "git status"
    assert calls[0][1][0]["_online_target"] == "git status"


def test_fine_tune_checkpoint_updates_model_and_preserves_config_and_vocab(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    save_test_checkpoint(checkpoint, text="git status\npytest\n", seq_len=32)
    before = torch.load(checkpoint, map_location="cpu")

    cli.fine_tune_checkpoint(
        checkpoint,
        [{"action": "accepted", "command": "git status"}],
        epochs=1,
        batch_size=8,
        lr=1e-4,
        grad_clip=1.0,
    )

    after = torch.load(checkpoint, map_location="cpu")
    assert after["config"] == before["config"]
    assert after["vocab"] == before["vocab"]
    assert any(
        not torch.equal(before["model"][name], after["model"][name])
        for name in before["model"]
    )
    assert after["checkpoint_version"] == cli.CURRENT_CHECKPOINT_VERSION
    assert after["metadata"]["updated_by"] == "online-learn"
    assert after["metadata"]["migrated_from_version"] == 0


def test_fine_tune_checkpoint_updates_from_rejected_feedback(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    save_test_checkpoint(checkpoint, text="git status\nrm -rf .\n", seq_len=32)
    before = torch.load(checkpoint, map_location="cpu")

    cli.fine_tune_checkpoint(
        checkpoint,
        [{"action": "rejected", "prompt": "rm ", "suggestion": "-rf .", "reward": -1.0}],
        epochs=1,
        batch_size=8,
        lr=1e-4,
        grad_clip=1.0,
    )

    after = torch.load(checkpoint, map_location="cpu")
    assert any(
        not torch.equal(before["model"][name], after["model"][name])
        for name in before["model"]
    )


def test_fine_tune_checkpoint_handles_mixed_reward_batch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    save_test_checkpoint(checkpoint, text="git status\npytest\nrm -rf .\n", seq_len=32)

    cli.fine_tune_checkpoint(
        checkpoint,
        [
            {"action": "accepted", "command": "git status", "reward": 1.0},
            {"action": "edited", "command": "pytest", "reward": 0.5},
            {"action": "rejected", "prompt": "rm ", "suggestion": "-rf .", "reward": -1.0},
        ],
        epochs=1,
        batch_size=8,
        lr=1e-4,
        grad_clip=1.0,
    )


def test_fine_tune_checkpoint_with_same_seed_writes_identical_weights(tmp_path):
    source = tmp_path / "source.pt"
    save_test_checkpoint(source, text="git status\npytest\n", seq_len=32)
    payload = torch.load(source, map_location="cpu")
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(payload, first)
    torch.save(payload, second)
    events = [
        {"action": "accepted", "command": "git status", "reward": 1.0},
        {"action": "edited", "command": "pytest", "reward": 0.5},
    ]

    for checkpoint in (first, second):
        cli.fine_tune_checkpoint(
            checkpoint,
            events,
            epochs=1,
            batch_size=2,
            lr=1e-4,
            grad_clip=1.0,
            seed=321,
        )

    first_payload = torch.load(first, map_location="cpu")
    second_payload = torch.load(second, map_location="cpu")
    assert all(
        torch.equal(first_payload["model"][name], second_payload["model"][name])
        for name in first_payload["model"]
    )
    assert first_payload["metadata"]["seed"] == 321


def test_online_learn_cmd_updates_checkpoint_from_short_command_batch(tmp_path, capsys):
    model = tmp_path / "model" / "checkpoint.pt"
    model.parent.mkdir()
    save_test_checkpoint(model, text="a\n", seq_len=128)
    store = tmp_path / "feedback" / "events.jsonl"
    state = tmp_path / "feedback" / "online_state.json"
    write_jsonl(store, [{"action": "accepted", "command": "a"}])

    args = SimpleNamespace(
        model=str(model),
        store=str(store),
        state=str(state),
        min_events=1,
        max_events=8,
        epochs=1,
        batch_size=8,
        lr=1e-4,
        grad_clip=1.0,
    )

    cli.online_learn_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "model": str(model),
        "state": str(state),
        "trained_events": 1,
        "updated": True,
    }


def test_online_learn_returns_noop_when_no_usable_training_data(tmp_path, capsys):
    model = tmp_path / "model" / "checkpoint.pt"
    model.parent.mkdir()
    save_test_checkpoint(model)
    store = tmp_path / "feedback" / "events.jsonl"
    state = tmp_path / "feedback" / "online_state.json"
    write_jsonl(store, [{"action": "rejected", "command": "git status"}])

    args = SimpleNamespace(
        model=str(model),
        store=str(store),
        state=str(state),
        min_events=1,
        max_events=8,
        epochs=1,
        batch_size=8,
        lr=1e-4,
        grad_clip=1.0,
    )

    cli.online_learn_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["updated"] is False
    assert payload["trained_events"] == 0
    assert not state.exists()


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
