import json
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
