# AGENTS.md

## Project Overview

Shell RT is a local shell command prediction experiment. It trains a character-level LSTM from
local shell history, suggests command continuations, and records explicit feedback events for
future ranking or learning work.

## Development Notes

- Main CLI entry point: `shell_next_cmd_lstm.py`.
- Tests live under `tests/`.
- Run tests with:

```bash
.venv/bin/python -m pytest
```

- Feedback events are local user data and should stay untracked under `feedback/`.
- `suggest` should not auto-log feedback unless explicitly requested.
- Terminal integration is currently zsh-only and source-script based via `shell_rt.zsh`.
- Keep changes small and aligned with the current lightweight CLI structure.

## Current Roadmap Priority

1. Awareness of the current directory, git status, environment variables, open files, and exit
   codes.
2. Command safety checks before suggesting destructive commands.
3. Ranking multiple candidate commands.
4. Online learning while the terminal is being used.
5. Reinforcement learning from accepted, rejected, edited, or executed suggestions.
6. Model versioning, checkpoint metadata migration, and reproducible training seeds.
7. Packaging as an installable command-line tool.
