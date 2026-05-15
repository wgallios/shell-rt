import argparse
import json
import re

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple
from uuid import uuid4

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vocab.char_vocab import CharVocab
from dataset.char_dataset import CharDataset
from LSTM.CharLSTM import CharLSTM


DEFAULT_FEEDBACK_STORE = "./feedback/events.jsonl"
FEEDBACK_REWARDS = {
    "accepted": 1.0,
    "executed": 2.0,
    "edited": 0.5,
    "rejected": -1.0,
}


def read_shell_history() -> str:
    paths = [Path("~/.zsh_history").expanduser(), Path("~/.bash_history").expanduser()]
    lines: List[str] = []

    for p in paths:
        if not p.exists():
            continue

        with p.open("r", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")

                # zsh extended history format: ": 1627891234:0;git status"
                m = re.match(r"^: \d+:\d+;(.+)$", line)
                cmd = m.group(1) if m else line
                cmd = cmd.strip()

                if cmd:
                    lines.append(cmd)

    return "\n".join(lines) + "\n"





def train_model(args):
    text = read_shell_history()
    print(f"Read {len(text)} characters from shell history.")

    if len(text) < 200:
        print("Not enough data to train. Please add more commands to your shell history.")
        return

    vocab = CharVocab(text)
    dataset = CharDataset(text, vocab, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if len(dataset) == 0 or len(dataloader) == 0:
        print("Not enough command history for the requested sequence length/batch size.")
        print("Try lowering --seq-len or --batch-size, or collect more shell history.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharLSTM(
        vocab_size=len(vocab.itos),
        emb_dim=args.emb,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"epoch {epoch} | loss {avg_loss:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab.itos,
            "config": {
                "emb": args.emb,
                "hidden": args.hidden,
                "layers": args.layers,
                "dropout": args.dropout,
                "seq_len": args.seq_len,
            },
        },
        checkpoint_path,
    )
    print(f"Saved model to {checkpoint_path}")


@torch.no_grad()
def load_model(ckpt_path: Path) -> Tuple[CharLSTM, CharVocab, dict]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    itos = checkpoint["vocab"]
    vocab = CharVocab("")
    vocab.itos = itos
    vocab.stoi = {ch: i for i, ch in enumerate(itos)}

    config = checkpoint.get("config", {})
    model = CharLSTM(
        vocab_size=len(itos),
        emb_dim=config.get("emb", 128),
        hidden=config.get("hidden", 256),
        layers=config.get("layers", 2),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocab, config


@torch.no_grad()
def sample_next(
    model: CharLSTM,
    vocab: CharVocab,
    prompt: str,
    max_new: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 20,
    seq_len: int = 128,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ids = vocab.encode(prompt)
    if not ids:
        ids = [0]

    tensor_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated_ids: List[int] = []

    for _ in range(max_new):
        context = tensor_ids[:, -seq_len:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None and top_k > 0:
            values, indexes = torch.topk(logits, k=min(top_k, logits.size(-1)))
            probs = torch.softmax(values, dim=-1)
            next_id = indexes[0, torch.multinomial(probs[0], num_samples=1)]
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs[0], num_samples=1)

        tensor_ids = torch.cat([tensor_ids, next_id.reshape(1, 1)], dim=1)
        generated_ids.append(next_id.item())
        if vocab.itos[next_id.item()] == "\n":
            break

    return prompt + vocab.decode(generated_ids)


def suggest_cmd(args):
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        print(f"Model not found at {checkpoint_path}. Run `train` first.")
        return

    model, vocab, config = load_model(checkpoint_path)
    output = sample_next(
        model,
        vocab,
        prompt=args.prompt,
        max_new=args.max_new,
        temperature=args.temp,
        top_k=args.top_k,
        seq_len=config.get("seq_len", 128),
    )
    completion = output[len(args.prompt):].strip("\n")
    print(json.dumps({"prompt": args.prompt, "suggestion": completion}))


def build_feedback_event(args) -> dict[str, Any]:
    reward = FEEDBACK_REWARDS[args.action] if args.reward is None else args.reward
    command = args.command
    if command is None and args.action in {"accepted", "executed"}:
        command = args.suggestion

    event: dict[str, Any] = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prompt": args.prompt,
        "suggestion": args.suggestion,
        "action": args.action,
        "reward": float(reward),
        "cwd": str(Path.cwd()),
        "source": "cli",
    }

    if command is not None:
        event["command"] = command
    if args.exit_code is not None:
        event["exit_code"] = args.exit_code

    return event


def write_feedback_event(store: Path, event: dict[str, Any]) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def feedback_cmd(args):
    store = Path(args.store)
    event = build_feedback_event(args)
    write_feedback_event(store, event)
    print(json.dumps({"id": event["id"], "store": str(store), "reward": event["reward"]}))



def main():
    p = argparse.ArgumentParser(description="Train an LSTM model to predict the next shell command.")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--seq-len", type=int, default=128)
    t.add_argument("--emb", type=int, default=128)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--out-dir", type=str, default="./model")

    s = sub.add_parser("suggest")
    s.add_argument("--model", type=str, default="./model/checkpoint.pt")
    s.add_argument("--prompt", type=str, required=True, help="Seed text, e.g. 'git add .\\n'")
    s.add_argument("--max-new", type=int, default=120)
    s.add_argument("--temp", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=20)

    f = sub.add_parser("feedback")
    f.add_argument("--prompt", type=str, required=True, help="Original prompt text.")
    f.add_argument("--suggestion", type=str, required=True, help="Model suggestion text.")
    f.add_argument("--action", type=str, required=True, choices=sorted(FEEDBACK_REWARDS))
    f.add_argument("--command", type=str, default=None, help="Final command text, if any.")
    f.add_argument("--exit-code", type=int, default=None)
    f.add_argument("--reward", type=float, default=None, help="Override the default action reward.")
    f.add_argument("--store", type=str, default=DEFAULT_FEEDBACK_STORE)

    args = p.parse_args()

    if args.cmd == "train":
        train_model(args)
    elif args.cmd == "suggest":
        suggest_cmd(args)
    elif args.cmd == "feedback":
        feedback_cmd(args)
    else:
        p.print_help()



if __name__ == "__main__":
    main()
