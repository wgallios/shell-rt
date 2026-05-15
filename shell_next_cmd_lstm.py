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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_context_json(value: str) -> dict[str, Any]:
    try:
        context = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--context-json must be valid JSON: {exc.msg}") from exc

    if not isinstance(context, dict):
        raise argparse.ArgumentTypeError("--context-json must decode to a JSON object")

    return context


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


def collect_candidate_completions(
    sampler,
    *,
    attempts: int,
    prompt: str,
    max_new: int,
    temperature: float,
    top_k: int | None,
    seq_len: int,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for _ in range(attempts):
        output = sampler(
            prompt=prompt,
            max_new=max_new,
            temperature=temperature,
            top_k=top_k,
            seq_len=seq_len,
        )
        completion = output[len(prompt):].strip("\n")
        if not completion or completion in seen:
            continue
        seen.add(completion)
        candidates.append(completion)

    return candidates


def command_starts_with(command: str, prefixes: tuple[str, ...]) -> bool:
    normalized = command.strip()
    return any(normalized == prefix or normalized.startswith(f"{prefix} ") for prefix in prefixes)


def open_files_from_context(context: dict[str, Any]) -> list[str]:
    open_files = context.get("open_files")
    if not isinstance(open_files, list):
        return []

    return [path for path in open_files if isinstance(path, str) and path]


def context_open_file_score(command: str, open_files: list[str]) -> float:
    if not open_files:
        return 0.0

    score = 0.0
    normalized = command.strip()
    lower_normalized = normalized.lower()
    suffixes = {Path(path).suffix.lower() for path in open_files}

    for path in open_files:
        if path in normalized:
            score += 3.0
            continue

        basename = Path(path).name
        if basename and basename in normalized:
            score += 1.25

    if ".py" in suffixes and command_starts_with(normalized, ("pytest", "python -m pytest")):
        score += 1.0

    if suffixes.intersection({".js", ".jsx", ".ts", ".tsx"}) and command_starts_with(
        normalized,
        ("npm test", "npm run test", "pnpm test", "yarn test", "node"),
    ):
        score += 1.0

    if suffixes.intersection({".sh", ".zsh", ".bash"}) and (
        command_starts_with(normalized, ("zsh -n", "bash -n", "shellcheck"))
        or lower_normalized.startswith("sh -n ")
    ):
        score += 1.0

    return score


def context_score(command: str, context: dict[str, Any]) -> float:
    score = 0.0
    normalized = command.strip()
    git = context.get("git")
    env = context.get("env")
    cwd = context.get("cwd")

    if isinstance(git, dict):
        git_ref = git.get("ref")
        in_worktree = bool(git_ref) or any(git.get(key) is True for key in ("dirty", "untracked"))
        if in_worktree:
            if command_starts_with(normalized, ("git status", "git diff", "git log", "git branch", "git show")):
                score += 2.0
            elif command_starts_with(normalized, ("git",)):
                score += 0.75

        if git.get("dirty") is True or git.get("untracked") is True:
            if command_starts_with(normalized, ("git status", "git diff")):
                score += 3.0
            if command_starts_with(normalized, ("git add", "git commit")):
                score += 2.0
        elif in_worktree:
            if command_starts_with(normalized, ("git status", "git pull", "git log")):
                score += 1.25
            if command_starts_with(normalized, ("git add", "git commit")):
                score -= 0.5

    if isinstance(env, dict):
        has_python_env = bool(env.get("VIRTUAL_ENV")) or bool(env.get("PYENV_VERSION"))
        has_node_env = bool(env.get("NODE_ENV"))
        if has_python_env and command_starts_with(
            normalized,
            ("python", "python3", "pytest", "pip", "pip3"),
        ):
            score += 1.0
        if has_node_env and command_starts_with(normalized, ("npm", "pnpm", "yarn", "node")):
            score += 1.0

    if isinstance(cwd, str):
        cwd_name = Path(cwd).name.lower()
        if cwd_name in {"py", "python", "django", "flask"} and command_starts_with(
            normalized,
            ("python", "python3", "pytest", "pip", "pip3"),
        ):
            score += 0.5
        if cwd_name in {"node", "js", "javascript", "typescript", "ts", "react"} and command_starts_with(
            normalized,
            ("npm", "pnpm", "yarn", "node"),
        ):
            score += 0.5

    last_exit_code = context.get("last_exit_code")
    if isinstance(last_exit_code, int) and last_exit_code != 0:
        if command_starts_with(
            normalized,
            ("git status", "pytest", "python -m pytest", "npm test", "ls", "pwd"),
        ):
            score += 1.25

    score += context_open_file_score(normalized, open_files_from_context(context))

    return score


def choose_context_candidate(prompt: str, candidates: list[str], context: dict[str, Any]) -> str | None:
    if not candidates:
        return None

    return max(candidates, key=lambda candidate: context_score(prompt + candidate, context))


def suggest_cmd(args):
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        print(f"Model not found at {checkpoint_path}. Run `train` first.")
        return

    model, vocab, config = load_model(checkpoint_path)
    seq_len = config.get("seq_len", 128)
    if getattr(args, "context_json", None) is None:
        output = sample_next(
            model,
            vocab,
            prompt=args.prompt,
            max_new=args.max_new,
            temperature=args.temp,
            top_k=args.top_k,
            seq_len=seq_len,
        )
        completion = output[len(args.prompt):].strip("\n")
    else:
        sampler = lambda **kwargs: sample_next(model, vocab, **kwargs)
        candidates = collect_candidate_completions(
            sampler,
            attempts=getattr(args, "rank_candidates", 5),
            prompt=args.prompt,
            max_new=args.max_new,
            temperature=args.temp,
            top_k=args.top_k,
            seq_len=seq_len,
        )
        completion = choose_context_candidate(args.prompt, candidates, args.context_json)
        if completion is None:
            output = sample_next(
                model,
                vocab,
                prompt=args.prompt,
                max_new=args.max_new,
                temperature=args.temp,
                top_k=args.top_k,
                seq_len=seq_len,
            )
            completion = output[len(args.prompt):].strip("\n")

    payload: dict[str, Any] = {"prompt": args.prompt, "suggestion": completion}
    if getattr(args, "context_json", None) is not None:
        payload["context"] = args.context_json
    print(json.dumps(payload))


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
    if getattr(args, "context_json", None) is not None:
        event["context"] = args.context_json

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
    s.add_argument("--context-json", type=parse_context_json, default=None)
    s.add_argument("--rank-candidates", type=positive_int, default=5)

    f = sub.add_parser("feedback")
    f.add_argument("--prompt", type=str, required=True, help="Original prompt text.")
    f.add_argument("--suggestion", type=str, required=True, help="Model suggestion text.")
    f.add_argument("--action", type=str, required=True, choices=sorted(FEEDBACK_REWARDS))
    f.add_argument("--command", type=str, default=None, help="Final command text, if any.")
    f.add_argument("--exit-code", type=int, default=None)
    f.add_argument("--reward", type=float, default=None, help="Override the default action reward.")
    f.add_argument("--store", type=str, default=DEFAULT_FEEDBACK_STORE)
    f.add_argument("--context-json", type=parse_context_json, default=None)

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
