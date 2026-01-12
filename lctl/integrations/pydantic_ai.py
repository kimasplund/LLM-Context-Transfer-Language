"""PydanticAI integration for LCTL.

This module provides automatic tracing for PydanticAI agents.
It wraps the agent run methods to capture input/output and tool calls.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import wraps


from typing import Any, AsyncIterator, Callable, Dict, Optional, TypeVar, Union

from ..core.session import LCTLSession

try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.run import AgentRunResult

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False


class LCTLPydanticAITracer:
    """Tracer for PydanticAI agents."""

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        verbose: bool = False,
    ):
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "PydanticAI is not installed. Install with: pip install lctl[pydantic-ai]"
            )
        self.session = session or LCTLSession(chain_id=chain_id)
        self._verbose = verbose

    @property
    def chain(self):
        return self.session.chain

    def export(self, path: str):
        self.session.export(path)


class TracedAgent:
    """Wrapper for PydanticAI Agent with LCTL tracing."""


    def __init__(
        self,
        agent: Agent,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        verbose: bool = False,
    ):
        self.agent = agent
        self.tracer = LCTLPydanticAITracer(
            chain_id=chain_id, session=session, verbose=verbose
        )
        self._instrument_tools()

    def _instrument_tools(self):
        """Instrument agent tools to trace execution."""
        # PydanticAI v0.0.x / v1.x internal structure
        # Tools are stored in _function_toolset.tools
        if not hasattr(self.agent, "_function_toolset") or not self.agent._function_toolset:
            return

        if not hasattr(self.agent._function_toolset, "tools"):
            return

        for name, tool in self.agent._function_toolset.tools.items():
            if not hasattr(tool, "function"):
                continue
            
            # Avoid double instrumentation
            func = tool.function
            # Use strict check to handle Mocks
            if getattr(func, "_lctl_instrumented", False) is True:
                continue

            original_func = tool.function
            # Handle AsyncMock alongside normal coroutines
            is_async = asyncio.iscoroutinefunction(original_func) or \
                       (isinstance(original_func, MagicMock) and getattr(original_func, "_is_coroutine", False))

            if is_async:
                @wraps(original_func)
                async def wrapper(*args, **kwargs):
                    return await self._trace_tool_execution(name, original_func, *args, **kwargs)
            else:
                @wraps(original_func)
                def wrapper(*args, **kwargs):
                    # For sync functions, we can't await, but _trace_tool_execution might need to be adapted
                    # However, PydanticAI tools seem to be normalized to async often?
                    # Let's assume sync for now if not async.
                    # We need a sync version of _trace_tool_execution logic
                    return self._trace_tool_execution_sync(name, original_func, *args, **kwargs)

            wrapper._lctl_instrumented = True
            tool.function = wrapper
            
            # Also update function_schema.function if present
            if hasattr(tool, "function_schema") and hasattr(tool.function_schema, "function"):
                tool.function_schema.function = wrapper

    async def _trace_tool_execution(self, name: str, func: Callable, *args, **kwargs):
        """Trace execution of an async tool."""
        # Extract input summary
        # args[0] might be RunContext if takes_ctx=True
        input_data = str(args) + str(kwargs)
        
        # We don't have a distinct "step_start" for tool calls in LCTL usually, 
        # usually they are attached to steps or we use tool_call event.
        # But we need duration, so we track time here.
        import time
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration_ms = int((time.time() - start_time) * 1000)
            
            self.tracer.session.tool_call(
                tool=name,
                input_data=str(input_data)[:200],
                output_data=str(result)[:500],
                duration_ms=duration_ms
            )
            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.tracer.session.tool_call(
                tool=name,
                input_data=str(input_data)[:200],
                output_data=f"Error: {str(e)}",
                duration_ms=duration_ms
            )
            raise

    def _trace_tool_execution_sync(self, name: str, func: Callable, *args, **kwargs):
        """Trace execution of a sync tool."""
        input_data = str(args) + str(kwargs)
        import time
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration_ms = int((time.time() - start_time) * 1000)
            
            self.tracer.session.tool_call(
                tool=name,
                input_data=str(input_data)[:200],
                output_data=str(result)[:500],
                duration_ms=duration_ms
            )
            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.tracer.session.tool_call(
                tool=name,
                input_data=str(input_data)[:200],
                output_data=f"Error: {str(e)}",
                duration_ms=duration_ms
            )
            raise

    async def run(
        self,
        *args,
        **kwargs,
    ) -> AgentRunResult:
        """Run the agent with tracing."""
        # Check signature of agent.run
        # Typically run(user_prompt, ...)
        user_prompt = args[0] if args else kwargs.get("user_prompt", "")
        agent_name = getattr(self.agent, "name", "pydantic_ai_agent") or "pydantic_ai_agent"

        self.tracer.session.step_start(
            agent=agent_name,
            intent="run",
            input_summary=str(user_prompt)[:200],
        )

        try:
            result = await self.agent.run(*args, **kwargs)
            
            # Extract output
            # result.data exists for structured results, result.output might be for unstructured?
            # Adjust based on observed structure
            if hasattr(result, "data"):
                output_summary = str(result.data)
            elif hasattr(result, "output"):
                output_summary = str(result.output)
            else:
                output_summary = str(result)
            
            # Extract usage if available
            tokens_in = 0
            tokens_out = 0
            
            # Currently PydanticAI might expose usage in different ways
            # Check result.usage() method or attribute
            if hasattr(result, "usage"):
                usage = result.usage()
                tokens_in = usage.request_tokens or 0
                tokens_out = usage.response_tokens or 0

            self.tracer.session.step_end(
                outcome="success",
                output_summary=output_summary,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

            # Record LLM Trace
            # Capture full message history
            if hasattr(result, "all_messages"):
                messages = []
                try:
                    for msg in result.all_messages():
                        # PydanticAI messages are Pydantic models
                        if hasattr(msg, "model_dump"):
                            messages.append(msg.model_dump(mode="json"))
                        else:
                            messages.append(str(msg))
                    
                    self.tracer.session.llm_trace(
                        messages=messages,
                        response=output_summary,
                        model=getattr(self.agent, "model_name", "unknown"),
                        usage={"input": tokens_in, "output": tokens_out},
                    )
                except Exception:
                    pass

            return result

        except Exception as e:
            self.tracer.session.error(
                category="execution_error",
                error_type=type(e).__name__,
                message=str(e),
                recoverable=False,
            )
            self.tracer.session.step_end(outcome="error")
            raise

    @asynccontextmanager
    async def run_stream(self, *args, **kwargs):
        """Run the agent with streaming and tracing."""
        user_prompt = args[0] if args else kwargs.get("user_prompt", "")
        agent_name = getattr(self.agent, "name", "pydantic_ai_agent") or "pydantic_ai_agent"

        self.tracer.session.step_start(
            agent=agent_name,
            intent="run_stream",
            input_summary=str(user_prompt)[:200],
        )

        try:
            async with self.agent.run_stream(*args, **kwargs) as result:
                yield TracedStreamedRunResult(result, self.tracer, agent_name)
        except Exception as e:
            self.tracer.session.error(
                category="execution_error",
                error_type=type(e).__name__,
                message=str(e),
                recoverable=False,
            )
            self.tracer.session.step_end(outcome="error")
            raise


__all__ = [
    "LCTLPydanticAITracer",
    "TracedAgent",
    "trace_agent",
    "is_available",
]

class TracedStreamedRunResult:
    """Wrapper for StreamedRunResult to trace streaming."""

    def __init__(self, result: Any, tracer: LCTLPydanticAITracer, agent_name: str):
        self._result = result
        self._tracer = tracer
        self._agent_name = agent_name
        self._accumulated_content = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)

    async def stream_text(self, *args, **kwargs) -> AsyncIterator[str]:
        """Trace stream_text iteration."""
        # Record stream start
        # Note: LCTL doesn't have explicit STREAM_START event yet in session public API?
        # It's an internal event type.
        # We'll rely on the start of the step having been recorded in run_stream.
        
        async for chunk in self._result.stream_text(*args, **kwargs):
            self._accumulated_content += chunk
            # TODO: Emit stream chunk event if LCTL supports it
            # self._tracer.session.emit_stream_chunk(...)
            yield chunk
        
        # Stream finished
        self._finalize_trace()

    def _finalize_trace(self):
        """Record final step end with usage."""
        try:
            usage = self._result.usage()
            tokens_in = usage.request_tokens or 0
            tokens_out = usage.response_tokens or 0
        except Exception:
            tokens_in = 0
            tokens_out = 0

        self._tracer.session.step_end(
            outcome="success",
            output_summary=self._accumulated_content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        # Record LLM Trace
        if hasattr(self._result, "all_messages"):
            messages = []
            try:
                for msg in self._result.all_messages():
                    if hasattr(msg, "model_dump"):
                        messages.append(msg.model_dump(mode="json"))
                    else:
                        messages.append(str(msg))
                
                self._tracer.session.llm_trace(
                    messages=messages,
                    response=self._accumulated_content,
                    model=getattr(self._tracer, "chain", {}).get("model", "unknown"), # Tricky to get model here
                    usage={"input": tokens_in, "output": tokens_out},
                )
            except Exception:
                pass

def trace_agent(
    agent: Agent,
    chain_id: Optional[str] = None,
    session: Optional[LCTLSession] = None,
    verbose: bool = False,
) -> TracedAgent:
    """Wrap a PydanticAI agent with tracing."""
    return TracedAgent(agent, chain_id, session, verbose)

def is_available() -> bool:
    """Check if PydanticAI is available."""
    return PYDANTIC_AI_AVAILABLE
