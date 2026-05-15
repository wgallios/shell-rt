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
