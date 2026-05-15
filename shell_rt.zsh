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
typeset -g SHELL_RT_ONLINE_LEARNING="${SHELL_RT_ONLINE_LEARNING:-0}"
typeset -g SHELL_RT_ONLINE_STATE="${SHELL_RT_ONLINE_STATE:-$SHELL_RT_ROOT/feedback/online_state.json}"
typeset -g SHELL_RT_ONLINE_MIN_EVENTS="${SHELL_RT_ONLINE_MIN_EVENTS:-1}"
typeset -g SHELL_RT_ONLINE_MAX_EVENTS="${SHELL_RT_ONLINE_MAX_EVENTS:-8}"
typeset -g SHELL_RT_ONLINE_EPOCHS="${SHELL_RT_ONLINE_EPOCHS:-1}"
typeset -g SHELL_RT_ONLINE_BATCH_SIZE="${SHELL_RT_ONLINE_BATCH_SIZE:-8}"
typeset -g SHELL_RT_ONLINE_LR="${SHELL_RT_ONLINE_LR:-1e-4}"
typeset -g SHELL_RT_ONLINE_GRAD_CLIP="${SHELL_RT_ONLINE_GRAD_CLIP:-1.0}"

typeset -g SHELL_RT_SUGGESTION=""
typeset -g SHELL_RT_SUGGESTION_PROMPT=""
typeset -g SHELL_RT_SUGGESTION_CONTEXT=""
typeset -g SHELL_RT_LAST_EXIT_CODE=0
typeset -ga SHELL_RT_OPEN_FILES

function shell_rt_clear_suggestion() {
  SHELL_RT_SUGGESTION=""
  SHELL_RT_SUGGESTION_PROMPT=""
  SHELL_RT_SUGGESTION_CONTEXT=""
  POSTDISPLAY=""
}

function shell_rt_precmd() {
  local last_status=$?
  emulate -L zsh
  SHELL_RT_LAST_EXIT_CODE=$last_status
}

function shell_rt_track_open_file() {
  emulate -L zsh
  setopt no_aliases

  local path="$1"
  local -a next_files
  local existing

  [[ -f "$path" ]] || return 0
  path="${path:A}"

  next_files=("$path")
  for existing in "${SHELL_RT_OPEN_FILES[@]}"; do
    [[ "$existing" == "$path" ]] && continue
    next_files+=("$existing")
    (( ${#next_files} >= 5 )) && break
  done

  SHELL_RT_OPEN_FILES=("${next_files[@]}")
}

function shell_rt_preexec() {
  emulate -L zsh
  setopt no_aliases

  local command_line="$1"
  local -a words
  words=("${(z)command_line}")
  (( ${#words} > 0 )) || return 0

  local editor="${words[1]:t}"
  case "$editor" in
    vim|nvim|vi|nano|emacs|hx|code) ;;
    *) return 0 ;;
  esac

  local word previous
  local end_options=0
  for word in "${words[@]:1}"; do
    if (( ! end_options )); then
      if [[ "$word" == "--" ]]; then
        end_options=1
        previous="$word"
        continue
      fi
      if [[ "$word" == -* || "$word" == +* ]]; then
        previous="$word"
        continue
      fi
      if [[ "$previous" == "-c" || "$previous" == "--command" || "$previous" == "-u" ]]; then
        previous="$word"
        continue
      fi
    fi

    shell_rt_track_open_file "$word"
    previous="$word"
  done
}

autoload -Uz add-zsh-hook
add-zsh-hook precmd shell_rt_precmd
add-zsh-hook preexec shell_rt_preexec

function shell_rt_context_json() {
  emulate -L zsh
  setopt no_aliases

  local git_ref="" git_dirty="" git_untracked="" git_status="" line

  if command git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_ref="$(command git symbolic-ref --quiet --short HEAD 2>/dev/null || command git rev-parse --short HEAD 2>/dev/null)"
    git_status="$(command git status --porcelain 2>/dev/null)"
    for line in ${(f)git_status}; do
      if [[ "$line" == '?? '* ]]; then
        git_untracked="true"
      else
        git_dirty="true"
      fi
    done
  fi

  SHELL_RT_CONTEXT_CWD="$PWD" \
  SHELL_RT_CONTEXT_OLDPWD="${OLDPWD-}" \
  SHELL_RT_CONTEXT_LAST_EXIT_CODE="${SHELL_RT_LAST_EXIT_CODE:-0}" \
  SHELL_RT_CONTEXT_GIT_REF="$git_ref" \
  SHELL_RT_CONTEXT_GIT_DIRTY="$git_dirty" \
  SHELL_RT_CONTEXT_GIT_UNTRACKED="$git_untracked" \
  SHELL_RT_CONTEXT_OPEN_FILES="${(pj:\n:)SHELL_RT_OPEN_FILES}" \
  "$SHELL_RT_PYTHON" -c '
import json
import os

context = {
    "cwd": os.environ.get("SHELL_RT_CONTEXT_CWD", ""),
    "last_exit_code": int(os.environ.get("SHELL_RT_CONTEXT_LAST_EXIT_CODE") or 0),
}

oldpwd = os.environ.get("SHELL_RT_CONTEXT_OLDPWD")
if oldpwd:
    context["oldpwd"] = oldpwd

allowlist = ["SHELL", "TERM", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "PYENV_VERSION", "NODE_ENV"]
env = {name: os.environ[name] for name in allowlist if os.environ.get(name)}
if env:
    context["env"] = env

cwd = context["cwd"]
open_files = []
for path in os.environ.get("SHELL_RT_CONTEXT_OPEN_FILES", "").splitlines():
    if not path:
        continue
    try:
        if cwd and os.path.commonpath([os.path.abspath(cwd), os.path.abspath(path)]) == os.path.abspath(cwd):
            path = os.path.relpath(path, cwd)
    except ValueError:
        pass
    open_files.append(path)
if open_files:
    context["open_files"] = open_files[:5]

git_ref = os.environ.get("SHELL_RT_CONTEXT_GIT_REF")
if git_ref:
    context["git"] = {
        "ref": git_ref,
        "dirty": os.environ.get("SHELL_RT_CONTEXT_GIT_DIRTY") == "true",
        "untracked": os.environ.get("SHELL_RT_CONTEXT_GIT_UNTRACKED") == "true",
    }

print(json.dumps(context, sort_keys=True, separators=(",", ":")), end="")
' 2>/dev/null
}

function shell_rt_fetch_suggestion() {
  emulate -L zsh
  setopt no_aliases

  local prompt="$BUFFER"
  local context_json output suggestion
  local -a suggest_cmd

  shell_rt_clear_suggestion
  context_json="$(shell_rt_context_json)" || context_json='{}'

  suggest_cmd=(
    "$SHELL_RT_PYTHON"
    "$SHELL_RT_ROOT/shell_next_cmd_lstm.py"
    suggest
    --model "$SHELL_RT_MODEL"
    --prompt "$prompt"
    --context-json "$context_json"
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
  SHELL_RT_SUGGESTION_CONTEXT="$context_json"
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
  local context_json="$SHELL_RT_SUGGESTION_CONTEXT"

  BUFFER+="$suggestion"
  CURSOR=${#BUFFER}

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
    --context-json "$context_json"
  )

  if [[ "$SHELL_RT_ONLINE_LEARNING" == "1" ]]; then
    (
      "${feedback_cmd[@]}" >/dev/null 2>&1 && \
      "$SHELL_RT_PYTHON" "$SHELL_RT_ROOT/shell_next_cmd_lstm.py" online-learn \
        --model "$SHELL_RT_MODEL" \
        --store "$SHELL_RT_FEEDBACK_STORE" \
        --state "$SHELL_RT_ONLINE_STATE" \
        --min-events "$SHELL_RT_ONLINE_MIN_EVENTS" \
        --max-events "$SHELL_RT_ONLINE_MAX_EVENTS" \
        --epochs "$SHELL_RT_ONLINE_EPOCHS" \
        --batch-size "$SHELL_RT_ONLINE_BATCH_SIZE" \
        --lr "$SHELL_RT_ONLINE_LR" \
        --grad-clip "$SHELL_RT_ONLINE_GRAD_CLIP" >/dev/null 2>&1
    ) &!
  else
    "${feedback_cmd[@]}" >/dev/null 2>&1 &!
  fi

  shell_rt_clear_suggestion
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
