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
- Can return an opt-in ranked list of sampled candidate commands for CLI/API consumers.
- Applies deterministic safety checks that suppress clear destructive command suggestions before
  display.
- Prints suggestions as JSON so shell scripts or terminal integrations can consume them.
- Provides zsh and bash source scripts for manual inline suggestions in the command line.
- Appends local feedback events and rewards to `./feedback/events.jsonl` by default.
- Can opt in to reward-weighted online learning from explicit feedback events in small batches.
- Captures low-risk terminal context in the zsh integration, including current directory,
  previous exit code, allowlisted environment variables, concise git state, and recently opened
  editor files.

# Setup & Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
python shell_next_cmd_lstm.py --help
```

For local development as an installed command:

```bash
pip install -e .
shell-rt --help
```

Both forms remain supported. The examples below use the checkout script path; after installation,
you can replace `python shell_next_cmd_lstm.py` with `shell-rt`.

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

Callers that want to inspect alternatives can opt in to ranked candidate output:

```bash
python shell_next_cmd_lstm.py suggest \
  --prompt "git " \
  --context-json '{"git":{"ref":"main","dirty":true}}' \
  --include-candidates
```

With `--include-candidates`, the JSON response includes a `candidates` array. Each entry has:

- `completion`: the generated suffix for the prompt.
- `command`: the full command text, formed as `prompt + completion`.
- `score`: the ranking score. Context-aware calls use the same deterministic context heuristics;
  calls without context use `0.0` and preserve sampling order for ties.

The top-level `suggestion` is the first ranked candidate's `completion`. Without
`--include-candidates`, the default JSON contract remains unchanged and no `candidates` field is
emitted.

Before output, `suggest` applies a deterministic safety gate to the full proposed command text
formed from `--prompt` plus the generated completion. This v1 gate is conservative and suppresses
high-confidence destructive commands such as file deletion, forced git discard workflows, disk
formatting or device writes, power operations, recursive permission or ownership changes, and moves
to obvious trash paths. Suppressed suggestions keep the same JSON shape and return an empty
`"suggestion": ""`. Unsafe completions are filtered before ranking, so `--include-candidates`
never emits candidates that fail the safety gate.

# Zsh Inline Suggestions

Shell RT includes a lightweight zsh integration that you can source from `.zshrc`:

```zsh
source /path/to/shell-rt/shell_rt.zsh
```

When Shell RT is installed with `pip install -e .` or from a package, source the packaged script:

```zsh
source "$(shell-rt integration-path --shell zsh)"
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
SHELL_RT_DATA_ROOT=/path/to/shell-rt
SHELL_RT_PYTHON=/path/to/python
SHELL_RT_COMMAND="shell-rt"
SHELL_RT_MODEL=/path/to/checkpoint.pt
SHELL_RT_FEEDBACK_STORE=/path/to/events.jsonl
source /path/to/shell-rt/shell_rt.zsh
```

Online learning is disabled by default. To fine-tune the current checkpoint after accepted
suggestions from zsh, opt in before sourcing:

```zsh
SHELL_RT_ONLINE_LEARNING=1
SHELL_RT_ONLINE_MIN_EVENTS=1
SHELL_RT_ONLINE_MAX_EVENTS=8
source /path/to/shell-rt/shell_rt.zsh
```

When enabled, accepting a suggestion still writes feedback first. After that feedback command
succeeds, zsh starts `online-learn` in the background. The learner reads new explicit feedback
events from `SHELL_RT_FEEDBACK_STORE`, trains a tiny reward-weighted batch, writes a temporary
checkpoint, and atomically replaces `SHELL_RT_MODEL`. It uses `SHELL_RT_ONLINE_STATE` to track the
last processed byte offset. In v1, zsh only auto-logs accepted suggestions; rejected, edited, and
executed events are learned from when they are recorded through the `feedback` command or API.

# Bash Inline Suggestions

Shell RT also includes a sourceable bash integration:

```bash
source /path/to/shell-rt/shell_rt.bash
```

For installed usage:

```bash
source "$(shell-rt integration-path --shell bash)"
```

Bash uses Readline's `bind -x` support. Press `Ctrl-Space` to request a suggestion for the current
command line. Bash inserts the suggestion immediately at the cursor because Readline does not expose
the same ghost-text display mechanism used by zsh. The bash integration does not record accepted
feedback automatically; use the explicit `feedback` command for accepted, rejected, edited, or
executed events.

The bash integration uses the same environment variables as the zsh integration where applicable,
including `SHELL_RT_COMMAND`, `SHELL_RT_MODEL`, `SHELL_RT_FEEDBACK_STORE`, and the online-learning
settings reserved for parity.

# Checkpoints And Seeds

New checkpoints include `checkpoint_version: 1` plus metadata with UTC timestamps, writer identity,
and the optional training seed. The compatibility fields `model`, `vocab`, and `config` remain
unchanged. Older checkpoints without `checkpoint_version` still load as legacy version `0`, and
read-only commands such as `suggest` do not rewrite them.

To explicitly upgrade an older checkpoint in place:

```bash
python shell_next_cmd_lstm.py migrate-checkpoint --model ./model/checkpoint.pt
```

The migration command prints stable JSON with `updated`, `from_version`, `to_version`, and `model`.
Running it again on a current checkpoint is a no-op.

Training commands accept an optional seed:

```bash
python shell_next_cmd_lstm.py train --seed 1234
python shell_next_cmd_lstm.py online-learn --seed 1234
```

The seed is applied to Python and Torch RNGs, including CUDA RNGs when available, and shuffled
training loaders use a seeded Torch generator. Strict Torch deterministic algorithms are not
enabled, so GPU kernels may still vary across hardware, drivers, or Torch versions.

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

Feedback is stored as append-only JSONL. `suggest` never auto-logs feedback. Feedback collection by
itself does not update the model; model updates happen only through the explicit opt-in
`online-learn` command.

# Online Learning

Run reward-weighted online learning manually:

```bash
python shell_next_cmd_lstm.py online-learn \
  --model ./model/checkpoint.pt \
  --store ./feedback/events.jsonl \
  --state ./feedback/online_state.json \
  --rl-mode reward-weighted \
  --max-reward-abs 1.0
```

The command prints stable JSON:

```json
{"model": "model/checkpoint.pt", "state": "feedback/online_state.json", "trained_events": 1, "updated": true}
```

This v1 reinforcement-learning path is reward-weighted online fine-tuning, not a full
policy-gradient RL stack. Positive rewards reinforce target commands. Negative rewards, such as a
rejected suggestion, train against the proposed full command with a clamped negative weight.

Usable events are `accepted`, `executed`, `edited`, and `rejected` with non-zero rewards and usable
target text. `accepted` and `executed` use `command` when present and otherwise use `suggestion`;
`edited` requires `command`; `rejected` uses `prompt + suggestion`. Malformed events, missing
targets, unsupported actions, zero rewards, and rewards below `--min-reward` are ignored. Reward
magnitude is capped by `--max-reward-abs`.

If fewer than `--min-events` usable events are available, the command returns `"updated": false` and
does not advance the state offset. State advances only after a checkpoint update succeeds.

# Model Output

The model generates text, not validated shell commands. Suggestions should be treated as draft
completions that a user or shell integration reviews before execution. The safety gate is a
best-effort suppression layer for generated suggestions, not a general shell sandbox or command
validator.

Because this is a character model trained only on command history, output quality depends heavily
on the amount and consistency of available history. Small or noisy histories will produce weaker
suggestions.

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
