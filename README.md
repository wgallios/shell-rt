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
- Prints suggestions as JSON so shell scripts or terminal integrations can consume them.
- Provides a zsh source script for manual inline suggestions in the command line.
- Appends local feedback events and rewards to `./feedback/events.jsonl` by default.
- Captures low-risk terminal context in the zsh integration, including current directory,
  previous exit code, allowlisted environment variables, and concise git state.

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

Callers can attach structured context for logging and future ranking work:

```bash
python shell_next_cmd_lstm.py suggest \
  --prompt "git " \
  --context-json '{"cwd":"/tmp/project","last_exit_code":0}'
```

When provided, `suggest` echoes the context object in its JSON response. The model still generates
from `--prompt` only; context is collected and logged but is not yet used to improve generation.

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
model, train from rewards, or rank future suggestions yet.

# Model Output

The model generates text, not validated shell commands. Suggestions should be treated as draft
completions that a user or shell integration reviews before execution.

Because this is a character model trained only on command history, output quality depends heavily
on the amount and consistency of available history. Small or noisy histories will produce weaker
suggestions.

# Not Yet Implemented

- Using captured context for ranking or generation.
- Open-file awareness.
- Command safety checks before suggesting destructive commands.
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
