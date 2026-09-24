# AgentDojo ACF runner

Runner for [AgentDojo](https://github.com/ethz-spylab/agentdojo) as
the model-in-the-loop benchmark referenced throughout `evaluation.tex`'s
`%% TODO before submission` markers (End-to-End Attack Success, Utility
Preservation, Contribution of Multi-Surface Coverage, Attribution Between
Detection and Authorisation).

The runner wires `on_prompt` before the first model call, `on_tool_call`
before tool execution, and `on_context` after the tool returns. C0 has no
ACF hook. For the other conditions, the runner builds and starts a sidecar
from this checkout on a private socket with a fresh HMAC key. No end-to-end
result tables have been produced yet.

## Install

```sh
pip install 'agentdojo==0.1.35'
pip install -e sdk/python       # the ACF SDK itself
```

Needs Python >=3.10. Provider credentials (e.g. `OPENAI_API_KEY`) are only
needed for a cloud model. A local model works with no API key at all: point
`--model local` (or `--model vllm_parsed`) at an OpenAI-compatible server on
`localhost` via `--model-id <model served>` and, if the server isn't on port
8000, the `LOCAL_LLM_PORT` env var. `build_pipeline`/`get_llm` resolve
`"local"`/`"vllm_parsed"` the same way `AgentPipeline.from_config` does -
verified end to end against a real local Ollama server (see "Live-tested
against a local model" below). Offline tests work without credentials.
The managed sidecar uses a Unix socket, so this runner needs Linux or macOS.

`manifest.json` in this folder pins the exact AgentDojo version, license, and
the full list of task suites/attacks this runner was checked against -
verified by installing the real package into a clean venv and inspecting its
source (`inspect.getsource`/`inspect.signature`), not from memory or docs
alone. It's deliberately separate from `../manifest.json`, which is owned by
the InjecAgent/PINT runner's own re-pin workflow.

## What AgentDojo actually looks like (verified against agentdojo==0.1.35)

- Four task suites: `banking`, `slack`, `travel`, `workspace`. Default
  benchmark version `v1.2.2`.
- Attacks are named strings loaded via
  `agentdojo.attacks.attack_registry.load_attack(name, suite, pipeline)` -
  see `manifest.json` for the full list (`important_instructions`,
  `tool_knowledge`, `injecagent`, several DoS variants, etc.).
- A custom agent/defense is any `agentdojo.agent_pipeline.BasePipelineElement`
  subclass implementing:
  ```python
  def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args={}) \
      -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]: ...
  ```
- `AgentPipeline(elements)` just runs `.query()` on each element in sequence,
  threading the returned tuple forward. `AgentPipeline.from_config()` builds
  1 from a `PipelineConfig`, but its `defense=` field only accepts a name
  from AgentDojo's own fixed `DEFENSES` list (`tool_filter`,
  `transformers_pi_detector`, `repeat_user_prompt`,
  `spotlighting_with_delimiting`) - a custom defense like ACF can't go through
  that field, it raises `ValueError("Invalid defense name")`.
- So a custom defense has to be spliced into a manually-assembled element
  list instead. Reading `AgentPipeline.from_config`'s own source for its
  `transformers_pi_detector` case shows the exact splice point AgentDojo uses
  for a defense that inspects tool output before it goes back to the model:
  ```python
  tools_loop = ToolsExecutionLoop([ToolsExecutor(tool_output_formatter),
                                    <DEFENSE ELEMENT HERE>,
                                    llm])
  pipeline = AgentPipeline([system_message_component, init_query_component,
                            llm, tools_loop])
  ```
  That's the `on_context` enforcement point. `on_tool_call` checks the batch
  before execution through `ACFGuardedToolsExecutor`. An `on_prompt`
  element belongs earlier, between `init_query_component` and `llm`:
  ```python
  AgentPipeline([system_message_component, init_query_component,
                 <ON_PROMPT ELEMENT HERE>, llm, tools_loop])
  ```
- Per-task results come back as a `SuiteResults` dict with
  `utility_results`, `security_results` (keyed by `(user_task_id,
  injection_task_id)`), and `injection_tasks_utility_results` - structured
  data, not just printed output. `security_results[k] is True` means the
  attack succeeded for that pair. AgentDojo ignores selected injection task IDs
  for DoS attacks and always runs the suite's first injection task. The runner
  rejects `--injection-tasks` for these attacks and records the effective IDs.

## Conditions (C0-C5)

Defined in `conditions.py`, inferred from `evaluation.tex`'s own wording -
not yet agreed with the team, adjust freely:

| Condition | ACF hooks active | Source line in evaluation.tex |
|---|---|---|
| C0 | none (undefended baseline) | "run AgentDojo for C0 and C5" |
| C1 | `on_prompt` | "compare C1 prompt-only enforcement with C5 full ACF" |
| C3 | `on_context` | Content detection on tool output |
| C4 | `on_tool_call` | Tool checks before execution, including parameter scans |
| C5 | `on_prompt`, `on_context`, `on_tool_call` | All currently wired hooks |

C2 is intentionally absent - `evaluation.tex` never names it, so nothing is
invented to fill the gap.

## Files

- `conditions.py` - the `CONDITIONS` registry above.
- `acf_pipeline.py` - ACF hook elements and `build_pipeline(condition, llm,
  model_id=None)`. `llm` takes a `ModelsEnum` string (cloud),
  `"local"`/`"vllm_parsed"` (local server, pair with `model_id`), or a
  `BasePipelineElement` instance directly.
- `run_agentdojo.py` - CLI: `--condition --suite --attack --model
  [--model-id] [--benchmark-version] [--user-tasks ...]
  [--injection-tasks ...] [--acf-config path] [--out]`. It runs benign
  tasks and attacked tasks and records benign utility, utility under attack,
  and ASR. Existing output paths are never overwritten.
- `test_scaffold.py` - offline tests (no network, no API keys): condition
  registry, hook decisions, and `AgentPipeline` assembly with a dummy
  in-process LLM stand-in. Run with:
  ```sh
  python3 -m pytest benchmarks/agentdojo/test_scaffold.py -v
  ```

## Live-tested against a local model

C0 was run end to end against a real local model at zero cost: Ollama
(`ollama serve`) on `localhost:11434` serving `llama3.2:latest`, hit through
its OpenAI-compatible API (`provider="local"`, no API key). Two real bugs
turned up doing this, both fixed in this scaffold:

1. `agentdojo.benchmark.benchmark_suite_with_injections` (and the lower-level
   `run_task_with(out)_injection_tasks` it calls) reads `Logger.get()` to
   build a `TraceLogger`, but never opens a logger context itself. With
   nothing entered, `Logger.get()` falls back to a bare `NullLogger()` whose
   `.logdir` was never set (its own `__enter__` sets `.logdir` but, unlike
   the base `Logger.__enter__`, doesn't push onto `LOGGER_STACK`, so
   `Logger.get()` never sees it even inside `with NullLogger():`). Fixed by
   wrapping the call in `with OutputLogger(logdir=None):` instead, which
   does push correctly via the inherited base `__enter__`/`__exit__`.
2. Attacks that name-drop the model (`important_instructions` and its
   variants) call `get_model_name_from_pipeline(pipeline)`, which requires
   `pipeline.name` to contain 1 of `agentdojo.attacks.base_attacks.
   MODEL_NAMES`'s keys as a substring, a real model string, or literally
   `"local"`/`"vllm_parsed"`. A custom label like `"agentdojo-C0"` fails that
   lookup with a `ValueError`. Fixed by keeping the raw model identifier in
   `pipeline.name` (`f"{llm_name}-{condition.name}"`, e.g. `"local-C0"`).

With both fixed, a full run completed cleanly:

```sh
export LOCAL_LLM_PORT=11434
python3 benchmarks/agentdojo/run_agentdojo.py \
  --condition C0 --suite banking --attack important_instructions \
  --model local --model-id llama3.2:latest --user-tasks user_task_0 \
  --out /tmp/c0_result.json
```

`"status": "completed"`, all 9 `important_instructions` injection tasks run
against `user_task_0`, JSON result written with per-pair `utility_results`/
`security_results` plus aggregate rates. The previous C0 local run used
llama3.2, which failed the banking task. It confirms the runner reaches a
real model, but says nothing about ACF or a useful baseline for the paper.

## What hasn't been done

No run against the agreed cloud model or shared benchmark harness yet. C1/C3/C4/C5
have offline tests and a live-sidecar smoke test, but no model-in-the-loop
result or trace review. The default ACF allowlists include only demo tools.
C4/C5 stop before running if the selected config or its policy file excludes
normal tools in the AgentDojo suite. Prepare a separate benchmark config and
policy copy with the suite's legitimate tool names in both allowlists, then
pass its absolute path with `--acf-config`. The runner launches the sidecar
it builds with that config and stores hashes for the config and policy files.
Keep that config copy, its policies, and the source commit for the paper.

The context hook is a proxy for tool output because ACF v1 has no
`on_tool_result` hook. It sends the output with `rag` provenance, not
`tool_output` provenance. Tool errors are scanned with output text. SANITISE
replaces the output and clears the original error. The SDK decodes JSON-quoted
sanitised strings. A context BLOCK ends the agent turn before the
model sees the result. A blocked tool batch stops before execution. SANITISE
on tool arguments also aborts because the runner cannot safely parse
arbitrary sanitised text back into AgentDojo arguments. C4/C5 do not isolate
name-based authorisation when all legitimate suite tools are allowlisted;
record hook decisions separately before claiming which layer stopped attacks.
The conditions do not guarantee identical model trajectories. The string-based
`on_context` proxy cannot exercise the sidecar's structural context-size rule.
Keep both limits visible in paper comparisons.

The JSON includes the AgentDojo version, checkpoint and decoding settings,
checkout commit and local diff hash, plus the ACF config and policy hashes
for defended runs. AgentDojo's `local` backend picks a random seed for every
request, so the recorded settings alone do not make local runs deterministic.
Use repeated runs with a pinned model and config before reporting paper
numbers. The earlier llama3.2 C0 run was only a plumbing check.
