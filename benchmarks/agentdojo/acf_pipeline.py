"""ACF firewall spliced into an AgentDojo pipeline.

Verified against agentdojo==0.1.35 by importing the real package and
inspecting AgentPipeline.from_config's source directly (not from memory).
The insertion point that from_config uses for its own "transformers_pi_detector"
defense is:

    tools_loop = ToolsExecutionLoop([ToolsExecutor(tool_output_formatter),
                                      <DEFENSE ELEMENT HERE>,
                                      llm])
    pipeline = AgentPipeline([system_message_component, init_query_component,
                              llm, tools_loop])

That is the on_context enforcement point: it sees tool results before they
go back to the LLM. on_tool_call runs inside the guarded executor, before
the tool executes. An on_prompt element belongs earlier
in the outer list, between init_query_component and llm:

    AgentPipeline([system_message_component, init_query_component,
                   <ON_PROMPT ELEMENT HERE>, llm, tools_loop])

`PipelineConfig.defense` only accepts a name from agentdojo's own fixed
DEFENSES list (raises ValueError otherwise), so a custom defense can't go
through from_config's `defense=` field -- the pipeline has to be assembled
by hand as above. build_pipeline() below does that assembly.

An on_tool_call SANITISE response stops the batch because AgentDojo tool
arguments cannot safely be reconstructed from arbitrary sanitised text.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.functions_runtime import Env, EmptyEnv, FunctionsRuntime
from agentdojo.types import ChatMessage, text_content_block_from_string

from acf import Firewall
from acf.models import Decision, SanitiseResult

from conditions import CONDITIONS, Condition  # noqa: E402


class ACFPipelineElement(BasePipelineElement):
    """Wraps an ACF Firewall instance as an AgentDojo pipeline element.

    One instance is meant to cover exactly one ACF hook (on_prompt,
    on_context, or on_tool_call) -- construct one per active hook for a
    condition and splice each at its matching point (see module docstring),
    rather than one element trying to run every hook.
    """

    name = "acf_firewall"

    def __init__(self, firewall: Firewall, hook: str) -> None:
        if hook not in ("on_prompt", "on_context", "on_tool_call"):
            raise ValueError(f"Unknown ACF hook: {hook!r}")
        self._firewall = firewall
        self._hook = hook

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        args = extra_args if extra_args is not None else {}
        if self._hook == "on_prompt":
            if not messages or messages[-1]["role"] != "user":
                return query, runtime, env, messages, args
            result = self._firewall.on_prompt(query)
            if result == Decision.BLOCK:
                raise AbortAgentError("ACF blocked the prompt", list(messages), env)
            if isinstance(result, SanitiseResult):
                if result.sanitised_text is None:
                    raise AbortAgentError("ACF could not sanitise the prompt", list(messages), env)
                query = result.sanitised_text
                messages = [
                    *messages[:-1],
                    {**messages[-1], "content": [text_content_block_from_string(query)]},
                ]
        elif self._hook == "on_context":
            # ToolsExecutor has already appended these messages. Check only the
            # current batch, before the model sees it.
            start = len(messages)
            while start and messages[start - 1]["role"] == "tool":
                start -= 1
            if start == len(messages):
                return query, runtime, env, messages, args
            checked = list(messages)
            for index in range(start, len(messages)):
                message = messages[index]
                original = "\n".join(
                    block["content"] for block in message["content"]
                    if block["type"] == "text"
                )
                if message.get("error") is not None:
                    original = "\n".join((original, str(message["error"])))
                result = self._firewall.on_context([original])[0]
                if result.decision == Decision.ALLOW:
                    continue
                if result.decision == Decision.BLOCK or result.sanitised_text is None:
                    # The model must not get another turn after a blocked
                    # tool result. Scrub the whole batch from the trace too.
                    safe_messages = [
                        *messages[:start],
                        *(
                            {**tool, "content": [text_content_block_from_string("[ACF blocked tool output]")], "error": None}
                            for tool in messages[start:]
                        ),
                    ]
                    raise AbortAgentError("ACF blocked the tool output", safe_messages, env)
                checked[index] = {
                    **message,
                    "content": [text_content_block_from_string(result.sanitised_text)],
                    "error": None,
                }
            messages = checked
        return query, runtime, env, messages, args


class ACFGuardedToolsExecutor(ToolsExecutor):
    """Reject the whole tool batch before executing any denied call."""

    def __init__(self, firewall: Firewall) -> None:
        super().__init__()
        self._firewall = firewall

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args=None):
        if messages and messages[-1]["role"] == "assistant":
            for call in messages[-1].get("tool_calls") or []:
                result = self._firewall.on_tool_call(call.function, dict(call.args))
                if result != Decision.ALLOW:
                    # AgentDojo derives its action trace from assistant messages.
                    # Remove this batch since no tool has run yet.
                    safe_messages = [*messages[:-1], {**messages[-1], "tool_calls": []}]
                    raise AbortAgentError("ACF blocked the tool call", safe_messages, env)
        return super().query(query, runtime, env, messages, extra_args or {})


def build_pipeline(
    condition: Condition,
    llm: str | BasePipelineElement,
    model_id: str | None = None,
    firewall: Firewall | None = None,
) -> AgentPipeline:
    """Assemble an AgentPipeline for one condition.

    llm can be:
      - a BasePipelineElement instance -- used as-is (what the offline tests
        in test_scaffold.py pass, and also how to point at a local model
        server directly, e.g. LocalLLM(openai.OpenAI(base_url=...), model_id))
      - a ModelsEnum string value (e.g. "gpt-4o-mini-2024-07-18") -- resolved
        through AgentDojo's own get_llm(), same as AgentPipeline.from_config
        does, and needs the matching provider's API key set (e.g.
        OPENAI_API_KEY)
      - the literal string "local" or "vllm_parsed" -- resolved the same way,
        but talks to an OpenAI-compatible server on localhost instead of a
        cloud provider. No API key needed; needs model_id (the model name
        the local server serves) and, for "local", the LOCAL_LLM_PORT env var
        if the server isn't on the default port 8000. Verified end to end
        against a local Ollama server serving llama3.2 on port 11434 --
        see README.md.

    Conditions with ACF hooks require a live sidecar and ACF_HMAC_KEY.
    """
    if condition.acf_hooks and firewall is None:
        firewall = Firewall()

    from agentdojo.agent_pipeline.agent_pipeline import get_llm, load_system_message
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    from agentdojo.models import MODEL_PROVIDERS, ModelsEnum

    system_message_component = SystemMessage(load_system_message(None))
    init_query_component = InitQuery()

    if isinstance(llm, str):
        provider = MODEL_PROVIDERS[ModelsEnum(llm)]
        llm_element: BasePipelineElement = get_llm(provider, llm, model_id, tool_delimiter="tool")
        llm_name = llm
    else:
        llm_element = llm
        llm_name = llm.name

    tool_executor = (
        ACFGuardedToolsExecutor(firewall)
        if "on_tool_call" in condition.acf_hooks
        else ToolsExecutor(tool_result_to_str)
    )
    loop_elements = [tool_executor]
    if "on_context" in condition.acf_hooks:
        loop_elements.append(ACFPipelineElement(firewall, "on_context"))
    loop_elements.append(llm_element)
    tools_loop = ToolsExecutionLoop(loop_elements)
    elements = [system_message_component, init_query_component]
    if "on_prompt" in condition.acf_hooks:
        elements.append(ACFPipelineElement(firewall, "on_prompt"))
    elements.extend((llm_element, tools_loop))
    pipeline = AgentPipeline(
        elements
    )
    # Some attacks (e.g. important_instructions) call
    # get_model_name_from_pipeline(pipeline), which requires pipeline.name to
    # contain one of agentdojo.attacks.base_attacks.MODEL_NAMES' keys as a
    # substring (a real model string, or "local"/"vllm_parsed") -- a purely
    # custom label like "agentdojo-C0" fails that lookup. Keep the raw model
    # identifier in the name and append the condition for readability.
    pipeline.name = f"{llm_name}-{condition.name}"
    return pipeline


__all__ = ["ACFPipelineElement", "build_pipeline", "CONDITIONS"]
