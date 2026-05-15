import argparse
import json
import os
import random
import re
import shlex

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
DEFAULT_MODEL_PATH = "./model/checkpoint.pt"
DEFAULT_ONLINE_STATE = "./feedback/online_state.json"
DEFAULT_ONLINE_RL_MODE = "reward-weighted"
CURRENT_CHECKPOINT_VERSION = 1
FEEDBACK_REWARDS = {
    "accepted": 1.0,
    "executed": 2.0,
    "edited": 0.5,
    "rejected": -1.0,
}

COMMAND_SEPARATORS = {";", "&&", "||", "|"}
DESTRUCTIVE_COMMANDS = {"rm", "unlink", "rmdir", "shred", "wipefs"}
POWER_COMMANDS = {"shutdown", "reboot", "poweroff", "halt"}
SUDO_FLAGS_WITH_VALUES = {
    "-A",
    "-a",
    "-C",
    "-c",
    "-g",
    "-h",
    "-p",
    "-T",
    "-t",
    "-U",
    "-u",
    "--askpass",
    "--auth-type",
    "--close-from",
    "--group",
    "--host",
    "--prompt",
    "--user",
}
ENV_FLAGS_WITH_VALUES = {"-C", "-S", "-u", "--chdir", "--split-string", "--unset"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seed_rngs(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def checkpoint_version(checkpoint: dict[str, Any]) -> int:
    version = checkpoint.get("checkpoint_version", 0)
    if not isinstance(version, int):
        raise ValueError("Unsupported checkpoint version: checkpoint_version must be an integer")
    if version > CURRENT_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version}; "
            f"this shell-rt supports up to {CURRENT_CHECKPOINT_VERSION}"
        )
    if version < 0:
        raise ValueError(f"Unsupported checkpoint version {version}")
    return version


def checkpoint_metadata(
    *,
    created_by: str,
    updated_by: str | None = None,
    seed: int | None = None,
    migrated_from_version: int | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    metadata = dict(existing or {})
    metadata.setdefault("created_at", now)
    metadata.setdefault("created_by", created_by)
    metadata["updated_at"] = now
    metadata["updated_by"] = updated_by or created_by
    if seed is not None:
        metadata["seed"] = seed
    if migrated_from_version is not None:
        metadata["migrated_from_version"] = migrated_from_version
    return metadata


def make_checkpoint(
    *,
    model: dict[str, torch.Tensor],
    vocab: list[str],
    config: dict[str, Any],
    created_by: str,
    seed: int | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "vocab": vocab,
        "config": config,
        "checkpoint_version": CURRENT_CHECKPOINT_VERSION,
        "metadata": checkpoint_metadata(created_by=created_by, seed=seed),
    }


def ensure_current_checkpoint_for_write(
    checkpoint: dict[str, Any],
    *,
    updated_by: str,
    seed: int | None = None,
) -> dict[str, Any]:
    version = checkpoint_version(checkpoint)
    migrated_from_version = None if version == CURRENT_CHECKPOINT_VERSION else version
    existing_metadata = checkpoint.get("metadata")
    if not isinstance(existing_metadata, dict):
        existing_metadata = None

    checkpoint["checkpoint_version"] = CURRENT_CHECKPOINT_VERSION
    checkpoint["metadata"] = checkpoint_metadata(
        created_by=updated_by,
        updated_by=updated_by,
        seed=seed,
        migrated_from_version=migrated_from_version,
        existing=existing_metadata,
    )
    return checkpoint


def atomic_save_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    tmp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid4().hex}.tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def migrate_checkpoint_file(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    from_version = checkpoint_version(checkpoint)
    updated = from_version != CURRENT_CHECKPOINT_VERSION

    if updated:
        ensure_current_checkpoint_for_write(checkpoint, updated_by="migrate-checkpoint")
        atomic_save_checkpoint(checkpoint, checkpoint_path)

    return {
        "updated": updated,
        "from_version": from_version,
        "to_version": CURRENT_CHECKPOINT_VERSION,
        "model": str(checkpoint_path),
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


def shell_tokens(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return re.findall(r"&&|\|\||[;&|]|[^\s;&|]+", command)


def split_command_segments(command: str) -> list[list[str]]:
    segments: list[list[str]] = []

    for line in command.splitlines():
        segment: list[str] = []
        for token in shell_tokens(line):
            if token in COMMAND_SEPARATORS:
                if segment:
                    segments.append(segment)
                    segment = []
                continue
            segment.append(token)
        if segment:
            segments.append(segment)

    return segments


def is_assignment(token: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", token) is not None


def skip_option(tokens: list[str], index: int, flags_with_values: set[str]) -> int:
    token = tokens[index]
    if token == "--":
        return index + 1
    if token.startswith("--") and "=" in token:
        return index + 1
    if token in flags_with_values:
        return min(index + 2, len(tokens))
    return index + 1


def normalize_command_segment(tokens: list[str]) -> list[str]:
    normalized = list(tokens)

    while normalized:
        while normalized and is_assignment(normalized[0]):
            normalized = normalized[1:]

        if not normalized:
            return []

        command_name = Path(normalized[0]).name

        if command_name in {"command", "builtin"}:
            normalized = normalized[1:]
            continue

        if command_name == "sudo":
            index = 1
            while index < len(normalized) and normalized[index].startswith("-"):
                next_index = skip_option(normalized, index, SUDO_FLAGS_WITH_VALUES)
                if normalized[index] == "--":
                    index = next_index
                    break
                index = next_index
            normalized = normalized[index:]
            continue

        if command_name == "env":
            index = 1
            while index < len(normalized):
                token = normalized[index]
                if is_assignment(token):
                    index += 1
                    continue
                if token.startswith("-"):
                    next_index = skip_option(normalized, index, ENV_FLAGS_WITH_VALUES)
                    if token == "--":
                        index = next_index
                        break
                    index = next_index
                    continue
                break
            normalized = normalized[index:]
            continue

        return normalized

    return []


def has_force_flag(args: list[str]) -> bool:
    return any(arg == "--force" or (arg.startswith("-") and not arg.startswith("--") and "f" in arg) for arg in args)


def has_recursive_flag(args: list[str]) -> bool:
    return any(
        arg == "--recursive" or (arg.startswith("-") and not arg.startswith("--") and "R" in arg)
        for arg in args
    )


def is_trash_path(path: str) -> bool:
    parts = [part.lower() for part in Path(path).parts]
    return any(part in {".trash", "trash"} for part in parts)


def is_destructive_segment(tokens: list[str]) -> bool:
    normalized = normalize_command_segment(tokens)
    if not normalized:
        return False

    command_name = Path(normalized[0]).name
    args = normalized[1:]

    if command_name in DESTRUCTIVE_COMMANDS or command_name in POWER_COMMANDS:
        return True

    if command_name == "git" and len(args) >= 2:
        if args[0] == "reset" and "--hard" in args[1:]:
            return True
        if args[0] == "clean" and has_force_flag(args[1:]):
            return True

    if command_name == "mkfs" or command_name.startswith("mkfs."):
        return True

    if command_name in {"fdisk", "parted"}:
        return True

    if command_name == "dd" and any(arg.startswith("of=/dev/") for arg in args):
        return True

    if command_name in {"chmod", "chown", "chgrp"} and has_recursive_flag(args):
        return True

    if command_name == "mv" and args and is_trash_path(args[-1]):
        return True

    return False


def is_destructive_command(command: str) -> bool:
    return any(is_destructive_segment(segment) for segment in split_command_segments(command))


def is_safe_suggestion(prompt: str, completion: str) -> bool:
    return not is_destructive_command(prompt + completion)


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
    seed = getattr(args, "seed", None)
    generator = seed_rngs(seed)
    text = read_shell_history()
    print(f"Read {len(text)} characters from shell history.")

    if len(text) < 200:
        print("Not enough data to train. Please add more commands to your shell history.")
        return

    vocab = CharVocab(text)
    dataset = CharDataset(text, vocab, seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )

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
        make_checkpoint(
            model=model.state_dict(),
            vocab=vocab.itos,
            config={
                "emb": args.emb,
                "hidden": args.hidden,
                "layers": args.layers,
                "dropout": args.dropout,
                "seq_len": args.seq_len,
            },
            created_by="train",
            seed=seed,
        ),
        checkpoint_path,
    )
    print(f"Saved model to {checkpoint_path}")


@torch.no_grad()
def load_model(ckpt_path: Path) -> Tuple[CharLSTM, CharVocab, dict]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_version(checkpoint)
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


def rank_candidate_commands(
    prompt: str,
    candidates: list[str],
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []

    for completion in candidates:
        if not is_safe_suggestion(prompt, completion):
            continue

        command = prompt + completion
        score = context_score(command, context) if context is not None else 0.0
        ranked.append({"completion": completion, "command": command, "score": score})

    return sorted(ranked, key=lambda candidate: candidate["score"], reverse=True)


def suggest_cmd(args):
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        print(f"Model not found at {checkpoint_path}. Run `train` first.")
        return

    model, vocab, config = load_model(checkpoint_path)
    seq_len = config.get("seq_len", 128)
    context = getattr(args, "context_json", None)
    include_candidates = bool(getattr(args, "include_candidates", False))
    ranked_candidates: list[dict[str, Any]] = []

    if context is None and not include_candidates:
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
        if not is_safe_suggestion(args.prompt, completion):
            completion = ""
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
        ranked_candidates = rank_candidate_commands(args.prompt, candidates, context)
        if ranked_candidates:
            completion = ranked_candidates[0]["completion"]
        elif candidates:
            completion = ""
        elif context is not None and not include_candidates:
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
            if not is_safe_suggestion(args.prompt, completion):
                completion = ""
        else:
            completion = ""

    payload: dict[str, Any] = {"prompt": args.prompt, "suggestion": completion}
    if context is not None:
        payload["context"] = context
    if include_candidates:
        payload["candidates"] = ranked_candidates
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


def read_online_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"offset": 0}

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"offset": 0}

    offset = state.get("offset", 0)
    if not isinstance(offset, int) or offset < 0:
        offset = 0
    return {"offset": offset}


def write_online_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_name(f".{state_path.name}.{uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, state_path)


def feedback_training_target(event: dict[str, Any]) -> str | None:
    action = event.get("action")

    if action in {"accepted", "executed"}:
        command = event.get("command")
        if isinstance(command, str) and command:
            return command
        suggestion = event.get("suggestion")
        if isinstance(suggestion, str) and suggestion:
            return suggestion
        return None

    if action == "edited":
        command = event.get("command")
        if isinstance(command, str) and command:
            return command
        return None

    if action == "rejected":
        prompt = event.get("prompt")
        suggestion = event.get("suggestion")
        if isinstance(prompt, str) and isinstance(suggestion, str) and suggestion:
            return prompt + suggestion
        return None

    return None


def feedback_training_example(
    event: dict[str, Any],
    *,
    min_reward: float,
    max_reward_abs: float,
) -> dict[str, Any] | None:
    reward = event.get("reward", FEEDBACK_REWARDS.get(str(event.get("action")), 0.0))
    if not isinstance(reward, (int, float)):
        return None
    reward = float(reward)
    if reward == 0.0 or abs(reward) < min_reward:
        return None

    target = feedback_training_target(event)
    if target is None:
        return None

    clamped_reward = max(-max_reward_abs, min(max_reward_abs, reward))
    if clamped_reward == 0.0:
        return None

    return {
        "target": target,
        "reward": clamped_reward,
        "action": event.get("action"),
    }


def read_new_feedback_training_events(
    store_path: Path,
    *,
    offset: int,
    max_events: int,
    min_reward: float = 0.0,
    max_reward_abs: float = 1.0,
) -> tuple[list[dict[str, Any]], int]:
    if not store_path.exists():
        return [], offset

    events: list[dict[str, Any]] = []
    current_offset = offset

    with store_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if offset > file_size:
            offset = 0
        f.seek(offset)
        current_offset = offset

        for raw_line in f:
            current_offset = f.tell()
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            if not isinstance(event, dict):
                continue
            example = feedback_training_example(
                event,
                min_reward=min_reward,
                max_reward_abs=max_reward_abs,
            )
            if example is not None:
                event = dict(event)
                event["_online_target"] = example["target"]
                event["_online_reward"] = example["reward"]
                events.append(event)
                if len(events) >= max_events:
                    break

    return events, current_offset


def read_new_accepted_events(
    store_path: Path,
    *,
    offset: int,
    max_events: int,
) -> tuple[list[dict[str, Any]], int]:
    if not store_path.exists():
        return [], offset

    events: list[dict[str, Any]] = []
    current_offset = offset

    with store_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if offset > file_size:
            offset = 0
        f.seek(offset)
        current_offset = offset

        for raw_line in f:
            current_offset = f.tell()
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            if not isinstance(event, dict):
                continue
            command = event.get("command")
            if event.get("action") == "accepted" and isinstance(command, str) and command:
                events.append(event)
                if len(events) >= max_events:
                    break

    return events, current_offset


class OnlineLearnLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(self.fd, str(os.getpid()).encode("utf-8"))
        self.acquired = True
        return True

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def online_training_examples(
    events: list[dict[str, Any]],
    *,
    min_reward: float,
    max_reward_abs: float,
) -> list[dict[str, Any]]:
    examples = []
    for event in events:
        target = event.get("_online_target")
        reward = event.get("_online_reward")
        if isinstance(target, str) and isinstance(reward, (int, float)):
            examples.append({"target": target, "reward": float(reward)})
            continue

        example = feedback_training_example(
            event,
            min_reward=min_reward,
            max_reward_abs=max_reward_abs,
        )
        if example is not None:
            examples.append(example)
    return examples


def fine_tune_checkpoint(
    checkpoint_path: Path,
    events: list[dict[str, Any]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    min_reward: float = 0.0,
    max_reward_abs: float = 1.0,
    seed: int | None = None,
) -> None:
    generator = seed_rngs(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ensure_current_checkpoint_for_write(checkpoint, updated_by="online-learn", seed=seed)
    model, vocab, config = load_model(checkpoint_path)
    saved_seq_len = int(config.get("seq_len", 128))
    examples = online_training_examples(
        events,
        min_reward=min_reward,
        max_reward_abs=max_reward_abs,
    )
    datasets = []
    for example in examples:
        text = f"{example['target']}\n"
        encoded = vocab.encode(text)
        seq_len = min(saved_seq_len, max(0, len(encoded) - 1))
        if seq_len < 1:
            continue
        dataset = CharDataset(text, vocab, seq_len=seq_len)
        if len(dataset) > 0:
            datasets.append((dataset, abs(float(example["reward"])), float(example["reward"]) < 0))

    if not datasets:
        raise ValueError("not enough online data for sequence training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for dataset, reward_weight, negative_reward in datasets:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                generator=generator,
            )
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                if negative_reward:
                    loss = -loss
                loss = loss * reward_weight
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

    checkpoint["model"] = model.cpu().state_dict()
    atomic_save_checkpoint(checkpoint, checkpoint_path)


def online_learn_result(updated: bool, trained_events: int, model: Path, state: Path) -> dict[str, Any]:
    return {
        "updated": updated,
        "trained_events": trained_events,
        "model": str(model),
        "state": str(state),
    }


def online_learn_cmd(args):
    model_path = Path(args.model)
    store_path = Path(args.store)
    state_path = Path(args.state)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")

    if not model_path.exists():
        print(json.dumps(online_learn_result(False, 0, model_path, state_path), sort_keys=True))
        return

    with OnlineLearnLock(lock_path) as acquired:
        if not acquired:
            print(json.dumps(online_learn_result(False, 0, model_path, state_path), sort_keys=True))
            return

        state = read_online_state(state_path)
        min_reward = max(0.0, float(getattr(args, "min_reward", 0.0)))
        max_reward_abs = max(0.0, float(getattr(args, "max_reward_abs", 1.0)))
        events, new_offset = read_new_feedback_training_events(
            store_path,
            offset=state["offset"],
            max_events=args.max_events,
            min_reward=min_reward,
            max_reward_abs=max_reward_abs,
        )

        if len(events) < args.min_events:
            print(json.dumps(online_learn_result(False, 0, model_path, state_path), sort_keys=True))
            return

        fine_tune_checkpoint(
            model_path,
            events,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            grad_clip=args.grad_clip,
            min_reward=min_reward,
            max_reward_abs=max_reward_abs,
            seed=getattr(args, "seed", None),
        )
        write_online_state(
            state_path,
            {
                "offset": new_offset,
                "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
        print(json.dumps(online_learn_result(True, len(events), model_path, state_path), sort_keys=True))


def migrate_checkpoint_cmd(args):
    result = migrate_checkpoint_file(Path(args.model))
    print(json.dumps(result, sort_keys=True))



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
    t.add_argument("--seed", type=int, default=None)

    s = sub.add_parser("suggest")
    s.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    s.add_argument("--prompt", type=str, required=True, help="Seed text, e.g. 'git add .\\n'")
    s.add_argument("--max-new", type=int, default=120)
    s.add_argument("--temp", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=20)
    s.add_argument("--context-json", type=parse_context_json, default=None)
    s.add_argument("--rank-candidates", type=positive_int, default=5)
    s.add_argument("--include-candidates", action="store_true")

    f = sub.add_parser("feedback")
    f.add_argument("--prompt", type=str, required=True, help="Original prompt text.")
    f.add_argument("--suggestion", type=str, required=True, help="Model suggestion text.")
    f.add_argument("--action", type=str, required=True, choices=sorted(FEEDBACK_REWARDS))
    f.add_argument("--command", type=str, default=None, help="Final command text, if any.")
    f.add_argument("--exit-code", type=int, default=None)
    f.add_argument("--reward", type=float, default=None, help="Override the default action reward.")
    f.add_argument("--store", type=str, default=DEFAULT_FEEDBACK_STORE)
    f.add_argument("--context-json", type=parse_context_json, default=None)

    o = sub.add_parser("online-learn")
    o.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    o.add_argument("--store", type=str, default=DEFAULT_FEEDBACK_STORE)
    o.add_argument("--state", type=str, default=DEFAULT_ONLINE_STATE)
    o.add_argument("--min-events", type=positive_int, default=1)
    o.add_argument("--max-events", type=positive_int, default=8)
    o.add_argument("--epochs", type=positive_int, default=1)
    o.add_argument("--batch-size", type=positive_int, default=8)
    o.add_argument("--lr", type=float, default=1e-4)
    o.add_argument("--grad-clip", type=float, default=1.0)
    o.add_argument("--min-reward", type=float, default=0.0)
    o.add_argument("--max-reward-abs", type=float, default=1.0)
    o.add_argument("--rl-mode", choices=[DEFAULT_ONLINE_RL_MODE], default=DEFAULT_ONLINE_RL_MODE)
    o.add_argument("--seed", type=int, default=None)

    m = sub.add_parser("migrate-checkpoint")
    m.add_argument("--model", type=str, required=True)

    args = p.parse_args()

    if args.cmd == "train":
        train_model(args)
    elif args.cmd == "suggest":
        suggest_cmd(args)
    elif args.cmd == "feedback":
        feedback_cmd(args)
    elif args.cmd == "online-learn":
        online_learn_cmd(args)
    elif args.cmd == "migrate-checkpoint":
        migrate_checkpoint_cmd(args)
    else:
        p.print_help()



if __name__ == "__main__":
    main()
