# Source this file from bash to enable manual inline Shell RT suggestions.

if [[ -z ${BASH_VERSION-} ]]; then
  return 0 2>/dev/null || exit 0
fi

if [[ -z ${SHELL_RT_ROOT-} ]]; then
  SHELL_RT_SOURCE="${BASH_SOURCE[0]}"
  SHELL_RT_ROOT="$(cd -- "$(dirname -- "$SHELL_RT_SOURCE")" >/dev/null 2>&1 && pwd -P)"
fi
export SHELL_RT_ROOT

if [[ -z ${SHELL_RT_PYTHON-} ]]; then
  if [[ -x "$SHELL_RT_ROOT/.venv/bin/python" ]]; then
    SHELL_RT_PYTHON="$SHELL_RT_ROOT/.venv/bin/python"
  else
    SHELL_RT_PYTHON="python"
  fi
fi
export SHELL_RT_PYTHON

if [[ -z ${SHELL_RT_DATA_ROOT-} ]]; then
  if [[ -f "$SHELL_RT_ROOT/shell_next_cmd_lstm.py" ]]; then
    SHELL_RT_DATA_ROOT="$SHELL_RT_ROOT"
  else
    SHELL_RT_DATA_ROOT="$PWD"
  fi
fi
export SHELL_RT_DATA_ROOT

export SHELL_RT_MODEL="${SHELL_RT_MODEL:-$SHELL_RT_DATA_ROOT/model/checkpoint.pt}"
export SHELL_RT_FEEDBACK_STORE="${SHELL_RT_FEEDBACK_STORE:-$SHELL_RT_DATA_ROOT/feedback/events.jsonl}"
export SHELL_RT_ONLINE_LEARNING="${SHELL_RT_ONLINE_LEARNING:-0}"
export SHELL_RT_ONLINE_STATE="${SHELL_RT_ONLINE_STATE:-$SHELL_RT_DATA_ROOT/feedback/online_state.json}"
export SHELL_RT_ONLINE_MIN_EVENTS="${SHELL_RT_ONLINE_MIN_EVENTS:-1}"
export SHELL_RT_ONLINE_MAX_EVENTS="${SHELL_RT_ONLINE_MAX_EVENTS:-8}"
export SHELL_RT_ONLINE_EPOCHS="${SHELL_RT_ONLINE_EPOCHS:-1}"
export SHELL_RT_ONLINE_BATCH_SIZE="${SHELL_RT_ONLINE_BATCH_SIZE:-8}"
export SHELL_RT_ONLINE_LR="${SHELL_RT_ONLINE_LR:-1e-4}"
export SHELL_RT_ONLINE_GRAD_CLIP="${SHELL_RT_ONLINE_GRAD_CLIP:-1.0}"

SHELL_RT_LAST_EXIT_CODE=0
SHELL_RT_OPEN_FILES=()

shell_rt_cli_base() {
  if [[ -n ${SHELL_RT_COMMAND-} ]]; then
    # shellcheck disable=SC2206
    SHELL_RT_CLI_BASE=($SHELL_RT_COMMAND)
  elif [[ -f "$SHELL_RT_ROOT/shell_next_cmd_lstm.py" ]]; then
    SHELL_RT_CLI_BASE=("$SHELL_RT_PYTHON" "$SHELL_RT_ROOT/shell_next_cmd_lstm.py")
  else
    SHELL_RT_CLI_BASE=("shell-rt")
  fi
}

shell_rt_track_open_file() {
  local path="$1"
  local existing
  local next_files=()

  [[ -f "$path" ]] || return 0
  path="$(cd -- "$(dirname -- "$path")" >/dev/null 2>&1 && pwd -P)/$(basename -- "$path")"

  next_files=("$path")
  for existing in "${SHELL_RT_OPEN_FILES[@]}"; do
    [[ "$existing" == "$path" ]] && continue
    next_files+=("$existing")
    (( ${#next_files[@]} >= 5 )) && break
  done

  SHELL_RT_OPEN_FILES=("${next_files[@]}")
}

shell_rt_track_editor_command() {
  local command_line="$1"
  local words=()
  local editor word previous end_options

  read -r -a words <<< "$command_line"
  (( ${#words[@]} > 0 )) || return 0

  editor="$(basename -- "${words[0]}")"
  case "$editor" in
    vim|nvim|vi|nano|emacs|hx|code) ;;
    *) return 0 ;;
  esac

  end_options=0
  previous=""
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

shell_rt_prompt_command() {
  SHELL_RT_LAST_EXIT_CODE=$?
}

shell_rt_debug_trap() {
  shell_rt_track_editor_command "$BASH_COMMAND"
}

if [[ ${PROMPT_COMMAND-} != *shell_rt_prompt_command* ]]; then
  if [[ -n ${PROMPT_COMMAND-} ]]; then
    PROMPT_COMMAND="shell_rt_prompt_command; $PROMPT_COMMAND"
  else
    PROMPT_COMMAND="shell_rt_prompt_command"
  fi
fi
if [[ -z $(trap -p DEBUG) ]]; then
  trap 'shell_rt_debug_trap' DEBUG
fi

shell_rt_context_json() {
  local git_ref="" git_dirty="" git_untracked="" git_status line
  local open_files

  if command git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_ref="$(command git symbolic-ref --quiet --short HEAD 2>/dev/null || command git rev-parse --short HEAD 2>/dev/null)"
    git_status="$(command git status --porcelain 2>/dev/null)"
    while IFS= read -r line; do
      if [[ "$line" == '?? '* ]]; then
        git_untracked="true"
      elif [[ -n "$line" ]]; then
        git_dirty="true"
      fi
    done <<< "$git_status"
  fi

  printf -v open_files '%s\n' "${SHELL_RT_OPEN_FILES[@]}"
  SHELL_RT_CONTEXT_CWD="$PWD" \
  SHELL_RT_CONTEXT_OLDPWD="${OLDPWD-}" \
  SHELL_RT_CONTEXT_LAST_EXIT_CODE="${SHELL_RT_LAST_EXIT_CODE:-0}" \
  SHELL_RT_CONTEXT_GIT_REF="$git_ref" \
  SHELL_RT_CONTEXT_GIT_DIRTY="$git_dirty" \
  SHELL_RT_CONTEXT_GIT_UNTRACKED="$git_untracked" \
  SHELL_RT_CONTEXT_OPEN_FILES="$open_files" \
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

shell_rt_fetch_suggestion() {
  local prompt="$READLINE_LINE"
  local context_json output suggestion

  context_json="$(shell_rt_context_json)" || context_json='{}'
  shell_rt_cli_base

  output="$("${SHELL_RT_CLI_BASE[@]}" suggest \
    --model "$SHELL_RT_MODEL" \
    --prompt "$prompt" \
    --context-json "$context_json" 2>/dev/null)" || return 0

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
  )" || return 0

  [[ -n "$suggestion" ]] || return 0

  READLINE_LINE="${READLINE_LINE:0:READLINE_POINT}${suggestion}${READLINE_LINE:READLINE_POINT}"
  READLINE_POINT=$((READLINE_POINT + ${#suggestion}))
}

bind -x '"\C-@": shell_rt_fetch_suggestion'
