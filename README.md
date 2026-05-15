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

# Model Output

The model generates text, not validated shell commands. Suggestions should be treated as draft
completions that a user or shell integration reviews before execution.

Because this is a character model trained only on command history, output quality depends heavily
on the amount and consistency of available history. Small or noisy histories will produce weaker
suggestions.

# Not Yet Implemented

- Reinforcement learning from accepted, rejected, edited, or executed suggestions.
- Online learning while the terminal is being used.
- Terminal integration for inline suggestions or tab-completion behavior.
- Awareness of the current directory, git status, environment variables, open files, or command
  exit codes.
- Command safety checks before suggesting destructive commands.
- Ranking multiple candidate commands.
- A feedback store for user actions and rewards.
- Packaging as an installable command-line tool.
- Model versioning, checkpoint metadata migration, or reproducible training seeds.

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
