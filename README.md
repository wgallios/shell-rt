# Shell RT

Shell RT is an early shell command prediction experiment. The current implementation trains a
small character-level LSTM on local shell history and uses that model to suggest a continuation
for a prompt such as `git ` or `python `.

This is currently a supervised next-character prediction model. It learns from historical shell
commands in `~/.zsh_history` and `~/.bash_history`, saves a local checkpoint, and exposes a small
CLI for generating suggestions.

# What Works Today

- Reads zsh and bash history files from the current user account.
- Normalizes zsh extended history entries such as `: 1699999999:0;git status`.
- Builds a character vocabulary from observed command history.
- Trains a small PyTorch LSTM to predict the next character in command sequences.
- Saves model checkpoints under `./model/checkpoint.pt` by default.
- Loads a saved checkpoint and generates command continuations.
- Uses captured context for lightweight heuristic reranking when context is provided.
- Applies deterministic safety checks that suppress clear destructive command suggestions before
  display.
- Prints suggestions as JSON so shell scripts or terminal integrations can consume them.
- Provides a zsh source script for manual inline suggestions in the command line.
- Appends local feedback events and rewards to `./feedback/events.jsonl` by default.
- Captures low-risk terminal context in the zsh integration, including current directory,
  previous exit code, allowlisted environment variables, concise git state, and recently opened
  editor files.

# Setup & Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
python shell_next_cmd_lstm.py --help
```

# Usage

Train from `~/.zsh_history` and `~/.bash_history`:

```bash
python shell_next_cmd_lstm.py train --epochs 3
```

Generate a suggested continuation for a prompt:

```bash
python shell_next_cmd_lstm.py suggest --prompt "git "
```

The `suggest` command prints JSON so it can be consumed from shell scripts:

```json
{"prompt": "git ", "suggestion": "status"}
```

Callers can attach structured context for lightweight reranking:

```bash
python shell_next_cmd_lstm.py suggest \
  --prompt "git " \
  --context-json '{"cwd":"/tmp/project","last_exit_code":0}'
```

When provided, `suggest` echoes the context object in its JSON response. The LSTM still generates
from `--prompt` only; context is not model conditioning, retraining, or feedback learning. Instead,
`suggest` samples a small set of candidate continuations and chooses among them with deterministic
heuristics based on captured `cwd`, `last_exit_code`, `env`, `git`, and `open_files` fields. Use
`--rank-candidates` to control the candidate count, defaulting to `5`.

Before output, `suggest` applies a deterministic safety gate to the full proposed command text
formed from `--prompt` plus the generated completion. This v1 gate is conservative and suppresses
high-confidence destructive commands such as file deletion, forced git discard workflows, disk
formatting or device writes, power operations, recursive permission or ownership changes, and moves
to obvious trash paths. Suppressed suggestions keep the same JSON shape and return an empty
`"suggestion": ""`.

# Zsh Inline Suggestions

Shell RT includes a lightweight zsh integration that you can source from `.zshrc`:

```zsh
source /path/to/shell-rt/shell_rt.zsh
```

This v1 integration is manual-fetch, not automatic autosuggest-on-every-keystroke behavior.
Press `Ctrl-Space` to request a suggestion for the current command buffer. If the model returns
a suffix, zsh displays it inline as ghost text without inserting it. Press `Ctrl-F` to accept the
visible suffix; acceptance inserts the suffix and records an `accepted` feedback event.

The zsh integration sends the same context snapshot to `suggest` and to the accepted feedback
event. Environment capture is allowlist-only: `SHELL`, `TERM`, `VIRTUAL_ENV`, `CONDA_DEFAULT_ENV`,
`PYENV_VERSION`, and `NODE_ENV`.

It also tracks the last five regular files opened through terminal editor invocations such as
`vim`, `nvim`, `vi`, `nano`, `emacs`, `hx`, and `code`. This is recent editor-command awareness
only, not live editor integration. These paths are included as `open_files` in context JSON and are
used only to rerank sampled candidates, with boosts for commands that mention the file path,
mention the basename, or match common file-type workflows such as `pytest`, `npm test`, `node`,
`zsh -n`, and `shellcheck`.

The integration can be configured with zsh variables before sourcing the script:

```zsh
SHELL_RT_ROOT=/path/to/shell-rt
SHELL_RT_PYTHON=/path/to/python
SHELL_RT_MODEL=/path/to/checkpoint.pt
SHELL_RT_FEEDBACK_STORE=/path/to/events.jsonl
source /path/to/shell-rt/shell_rt.zsh
```

# Feedback Logging

Record explicit user feedback for a suggestion:

```bash
python shell_next_cmd_lstm.py feedback \
  --prompt "git " \
  --suggestion "status" \
  --action accepted \
  --context-json '{"cwd":"/tmp/project","last_exit_code":0}'
```

Rejected, edited, and executed suggestions can be logged too:

```bash
python shell_next_cmd_lstm.py feedback --prompt "git " --suggestion "status" --action rejected
python shell_next_cmd_lstm.py feedback --prompt "git " --suggestion "stat" --action edited --command "git status" --reward 0.75
python shell_next_cmd_lstm.py feedback --prompt "pytest " --suggestion "tests/" --action executed --exit-code 0
```

Feedback is stored as append-only JSONL. This is data collection only; it does not update the
model or train from rewards.

# Model Output

The model generates text, not validated shell commands. Suggestions should be treated as draft
completions that a user or shell integration reviews before execution. The safety gate is a
best-effort suppression layer for generated suggestions, not a general shell sandbox or command
validator.

Because this is a character model trained only on command history, output quality depends heavily
on the amount and consistency of available history. Small or noisy histories will produce weaker
suggestions.

# Not Yet Implemented

- Ranking multiple candidate commands.
- Online learning while the terminal is being used.
- Reinforcement learning from accepted, rejected, edited, or executed suggestions.
- Model versioning, checkpoint metadata migration, or reproducible training seeds.
- Packaging as an installable command-line tool.

# Testing

```bash
pip install -r requirements.txt
.venv/bin/python -m pytest
```

# Pre-requisites

Ensure you have python 3.12+ installed along with pip. Ubuntu packages you may need are:

```bash
sudo apt-get install python3.12-venv
```
