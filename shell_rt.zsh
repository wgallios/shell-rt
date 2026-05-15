# Source this file from zsh to enable manual inline Shell RT suggestions.

if [[ -z ${ZSH_VERSION-} ]]; then
  return 0 2>/dev/null || exit 0
fi

typeset -g SHELL_RT_ROOT="${SHELL_RT_ROOT:-${${(%):-%x}:A:h}}"

if [[ -z ${SHELL_RT_PYTHON-} ]]; then
  if [[ -x "$SHELL_RT_ROOT/.venv/bin/python" ]]; then
    typeset -g SHELL_RT_PYTHON="$SHELL_RT_ROOT/.venv/bin/python"
  else
    typeset -g SHELL_RT_PYTHON="python"
  fi
fi

typeset -g SHELL_RT_MODEL="${SHELL_RT_MODEL:-$SHELL_RT_ROOT/model/checkpoint.pt}"
typeset -g SHELL_RT_FEEDBACK_STORE="${SHELL_RT_FEEDBACK_STORE:-$SHELL_RT_ROOT/feedback/events.jsonl}"

typeset -g SHELL_RT_SUGGESTION=""
typeset -g SHELL_RT_SUGGESTION_PROMPT=""

function shell_rt_clear_suggestion() {
  SHELL_RT_SUGGESTION=""
  SHELL_RT_SUGGESTION_PROMPT=""
  POSTDISPLAY=""
}

function shell_rt_fetch_suggestion() {
  emulate -L zsh
  setopt no_aliases

  local prompt="$BUFFER"
  local output suggestion
  local -a suggest_cmd

  shell_rt_clear_suggestion

  suggest_cmd=(
    "$SHELL_RT_PYTHON"
    "$SHELL_RT_ROOT/shell_next_cmd_lstm.py"
    suggest
    --model "$SHELL_RT_MODEL"
    --prompt "$prompt"
  )

  output="$("${suggest_cmd[@]}" 2>/dev/null)" || {
    zle redisplay
    return 0
  }

  suggestion="$(
    SHELL_RT_JSON="$output" "$SHELL_RT_PYTHON" -c '
import json
import os

try:
    payload = json.loads(os.environ["SHELL_RT_JSON"])
except Exception:
    raise SystemExit(1)

suggestion = payload.get("suggestion", "")
if isinstance(suggestion, str):
    print(suggestion, end="")
' 2>/dev/null
  )" || {
    zle redisplay
    return 0
  }

  if [[ -z "$suggestion" ]]; then
    zle redisplay
    return 0
  fi

  SHELL_RT_SUGGESTION="$suggestion"
  SHELL_RT_SUGGESTION_PROMPT="$prompt"
  POSTDISPLAY="$suggestion"
  zle redisplay
}

function shell_rt_accept_suggestion() {
  emulate -L zsh
  setopt no_aliases

  if [[ -z "$SHELL_RT_SUGGESTION" || "$BUFFER" != "$SHELL_RT_SUGGESTION_PROMPT" ]]; then
    shell_rt_clear_suggestion
    zle redisplay
    return 0
  fi

  local prompt="$SHELL_RT_SUGGESTION_PROMPT"
  local suggestion="$SHELL_RT_SUGGESTION"

  BUFFER+="$suggestion"
  CURSOR=${#BUFFER}
  shell_rt_clear_suggestion

  local -a feedback_cmd
  feedback_cmd=(
    "$SHELL_RT_PYTHON"
    "$SHELL_RT_ROOT/shell_next_cmd_lstm.py"
    feedback
    --prompt "$prompt"
    --suggestion "$suggestion"
    --action accepted
    --command "$BUFFER"
    --store "$SHELL_RT_FEEDBACK_STORE"
  )
  "${feedback_cmd[@]}" >/dev/null 2>&1 &!

  zle redisplay
}

function shell_rt_zle_line_pre_redraw() {
  emulate -L zsh

  if [[ -n "$SHELL_RT_SUGGESTION" && "$BUFFER" != "$SHELL_RT_SUGGESTION_PROMPT" ]]; then
    shell_rt_clear_suggestion
  fi
}

function shell_rt_zle_line_finish() {
  emulate -L zsh
  shell_rt_clear_suggestion
}

zle -N shell_rt_fetch_suggestion
zle -N shell_rt_accept_suggestion
zle -N zle-line-pre-redraw shell_rt_zle_line_pre_redraw
zle -N zle-line-finish shell_rt_zle_line_finish

bindkey '^@' shell_rt_fetch_suggestion
bindkey '^F' shell_rt_accept_suggestion
