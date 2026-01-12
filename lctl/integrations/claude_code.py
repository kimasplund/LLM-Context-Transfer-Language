"""Claude Code integration for LCTL.

This module provides tracing for Claude Code's multi-agent workflows using
the Task tool. It captures agent spawning, tool calls, and agent completions
as LCTL events for debugging and analysis.

Usage with Claude Code hooks:

    1. Create hook scripts in your project's .claude/hooks/ directory
    2. Use the LCTLClaudeCodeTracer to record events

Example hook script (.claude/hooks/PostToolUse.sh):

    #!/bin/bash
    # Trace Task tool completions
    if [ "$CLAUDE_TOOL_NAME" = "Task" ]; then
        python -c "
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer
    import os, json

    tracer = LCTLClaudeCodeTracer.get_or_create()
    tracer.on_task_complete(
        agent_type=os.environ.get('CLAUDE_TOOL_INPUT_subagent_type', 'unknown'),
        description=os.environ.get('CLAUDE_TOOL_INPUT_description', ''),
        result=os.environ.get('CLAUDE_TOOL_RESULT', '')[:1000]
    )
    "
    fi

Programmatic usage:

    from lctl.integrations.claude_code import LCTLClaudeCodeTracer

    # Start tracing a workflow
    tracer = LCTLClaudeCodeTracer(chain_id="my-workflow")

    # Record agent spawning
    tracer.on_task_start(
        agent_type="implementor",
        description="Implement authentication",
        prompt="Add JWT auth to the API..."
    )

    # Record agent completion
    tracer.on_task_complete(
        agent_type="implementor",
        description="Implement authentication",
        result="Successfully implemented JWT authentication..."
    )

    # Record tool calls within agents
    tracer.on_tool_call(
        tool_name="Bash",
        input_data={"command": "pytest"},
        output_data={"exit_code": 0},
        agent="implementor"
    )

    # Export trace
    tracer.export("workflow-trace.lctl.json")
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.session import LCTLSession

# Singleton instance for hook-based usage
_tracer_instance: Optional["LCTLClaudeCodeTracer"] = None
_tracer_file: Optional[Path] = None


class LCTLClaudeCodeTracer:
    """Tracer for Claude Code multi-agent workflows.

    This tracer captures events from Claude Code's Task tool usage,
    enabling time-travel debugging of multi-agent workflows.

    Attributes:
        session: The underlying LCTL session
        agent_stack: Stack of currently active agents (for nested spawning)
        tool_counts: Count of tool calls per agent
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        session: Optional[LCTLSession] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize the Claude Code tracer.

        Args:
            chain_id: Identifier for the trace chain
            session: Existing LCTL session to use
            output_dir: Directory for trace output (default: .claude/traces/)
        """
        self.session = session or LCTLSession(
            chain_id=chain_id or f"claude-code-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        self.output_dir = Path(output_dir) if output_dir else Path(".claude/traces")
        self.agent_stack: List[Dict[str, Any]] = []
        self.tool_counts: Dict[str, int] = {}
        self._start_times: Dict[str, float] = {}

    @classmethod
    def get_or_create(
        cls,
        chain_id: Optional[str] = None,
        state_file: Optional[str] = None,
    ) -> "LCTLClaudeCodeTracer":
        """Get existing tracer instance or create new one.

        This is useful for hook scripts that need to maintain state
        across multiple invocations.

        Args:
            chain_id: Chain ID for new tracer
            state_file: File to persist tracer state

        Returns:
            Tracer instance
        """
        global _tracer_instance, _tracer_file

        state_path = Path(state_file) if state_file else Path(".claude/traces/.lctl-state.json")

        if _tracer_instance is not None:
            return _tracer_instance

        # Try to restore from state file
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)

                # Load existing chain
                chain_path = Path(state.get("chain_path", ""))
                if chain_path.exists():
                    from ..core.events import Chain
                    chain = Chain.load(chain_path)
                    session = LCTLSession(chain_id=chain.id)
                    session.chain = chain
                    session._seq = len(chain.events)

                    _tracer_instance = cls(session=session)
                    _tracer_instance.agent_stack = state.get("agent_stack", [])
                    _tracer_instance.tool_counts = state.get("tool_counts", {})
                    _tracer_instance._start_times = state.get("start_times", {})
                    _tracer_file = state_path
                    return _tracer_instance
            except Exception:
                pass  # Fall through to create new

        # Create new tracer
        _tracer_instance = cls(chain_id=chain_id)
        _tracer_file = state_path
        _tracer_instance._save_state()
        return _tracer_instance

    def _save_state(self) -> None:
        """Persist tracer state for hook continuity."""
        if _tracer_file is None:
            return

        _tracer_file.parent.mkdir(parents=True, exist_ok=True)

        # Save chain
        chain_path = self.output_dir / f"{self.session.chain.id}.lctl.json"
        chain_path.parent.mkdir(parents=True, exist_ok=True)
        self.session.export(str(chain_path))

        # Save state
        state = {
            "chain_path": str(chain_path),
            "agent_stack": self.agent_stack,
            "tool_counts": self.tool_counts,
            "start_times": self._start_times,
        }
        with open(_tracer_file, "w") as f:
            json.dump(state, f)

    def on_task_start(
        self,
        agent_type: str,
        description: str,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> None:
        """Record a Task tool invocation (agent spawn).

        Args:
            agent_type: The subagent_type parameter (e.g., "implementor", "Explore")
            description: The task description
            prompt: The full prompt sent to the agent
            model: Optional model override
        """
        self._start_times[agent_type] = time.time()

        # Track nested agents
        self.agent_stack.append({
            "agent_type": agent_type,
            "description": description,
            "start_time": self._start_times[agent_type],
        })

        self.session.step_start(
            agent=agent_type,
            intent=description[:100],
            input_summary=prompt[:500] if prompt else description,
        )

        # Add fact about agent spawn
        self.session.add_fact(
            fact_id=f"spawn-{agent_type}-{len(self.session.chain.events)}",
            text=f"Spawned {agent_type}: {description}",
            confidence=1.0,
            source="claude-code",
        )

        if model:
            self.session.add_fact(
                fact_id=f"model-{agent_type}-{len(self.session.chain.events)}",
                text=f"Using model: {model}",
                confidence=1.0,
                source="claude-code",
            )

        self._save_state()

    def on_task_complete(
        self,
        agent_type: str,
        description: str = "",
        result: str = "",
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a Task tool completion.

        Args:
            agent_type: The agent that completed
            description: Task description
            result: Result summary from the agent
            success: Whether the agent succeeded
            error_message: Error message if failed
        """
        duration_ms = 0
        if agent_type in self._start_times:
            duration_ms = int((time.time() - self._start_times[agent_type]) * 1000)
            del self._start_times[agent_type]

        # Pop from agent stack
        if self.agent_stack and self.agent_stack[-1]["agent_type"] == agent_type:
            self.agent_stack.pop()

        outcome = "success" if success else "failure"

        self.session.step_end(
            agent=agent_type,
            outcome=outcome,
            duration_ms=duration_ms,
        )

        # Add fact about result
        if result:
            self.session.add_fact(
                fact_id=f"result-{agent_type}-{len(self.session.chain.events)}",
                text=f"{agent_type} result: {result[:500]}",
                confidence=0.9 if success else 0.5,
                source=agent_type,
            )

        # Record error if failed
        if not success and error_message:
            self.session.error(
                category="agent_failure",
                error_type="TaskError",
                message=error_message,
                recoverable=True,
            )

        self._save_state()

    def on_tool_call(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0,
        agent: Optional[str] = None,
    ) -> None:
        """Record a tool call within an agent.

        Args:
            tool_name: Name of the tool (Bash, Read, Write, etc.)
            input_data: Tool input parameters
            output_data: Tool output
            duration_ms: Tool execution time
            agent: Agent that made the call (inferred from stack if not provided)
        """
        # Infer agent from stack
        if agent is None and self.agent_stack:
            agent = self.agent_stack[-1]["agent_type"]

        # Track tool counts
        key = f"{agent or 'main'}:{tool_name}"
        self.tool_counts[key] = self.tool_counts.get(key, 0) + 1

        # Truncate large data
        def truncate(data: Any, max_len: int = 500) -> Any:
            if isinstance(data, str):
                return data[:max_len] + "..." if len(data) > max_len else data
            elif isinstance(data, dict):
                return {k: truncate(v, max_len) for k, v in list(data.items())[:10]}
            return data

        self.session.tool_call(
            tool=tool_name,
            input_data=truncate(input_data),
            output_data=truncate(output_data or {}),
            duration_ms=duration_ms,
        )

        self._save_state()

    def on_fact_discovered(
        self,
        fact_id: str,
        text: str,
        confidence: float = 0.8,
        agent: Optional[str] = None,
    ) -> None:
        """Record a fact discovered by an agent.

        Args:
            fact_id: Unique fact identifier
            text: Fact content
            confidence: Confidence score (0.0-1.0)
            agent: Agent that discovered the fact
        """
        source = agent
        if source is None and self.agent_stack:
            source = self.agent_stack[-1]["agent_type"]

        self.session.add_fact(
            fact_id=fact_id,
            text=text,
            confidence=confidence,
            source=source or "claude-code",
        )
        self._save_state()

    def on_fact_updated(
        self,
        fact_id: str,
        confidence: Optional[float] = None,
        text: Optional[str] = None,
        reason: str = "",
    ) -> None:
        """Update a fact's confidence or text.

        Args:
            fact_id: Fact to update
            confidence: New confidence score
            text: New text (optional)
            reason: Reason for update
        """
        self.session.modify_fact(
            fact_id=fact_id,
            confidence=confidence,
            text=text,
            reason=reason,
        )
        self._save_state()

    def checkpoint(self, description: str = "") -> None:
        """Create a checkpoint for fast replay.

        Args:
            description: Checkpoint description
        """
        self.session.checkpoint(description=description)
        self._save_state()

    def export(self, path: Optional[str] = None) -> str:
        """Export the trace to a file.

        Args:
            path: Output path (default: auto-generated in output_dir)

        Returns:
            Path to exported file
        """
        if path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = str(self.output_dir / f"{self.session.chain.id}.lctl.json")

        self.session.export(path)
        return path

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the traced workflow.

        Returns:
            Summary dict with agent stats, tool counts, facts, etc.
        """
        from ..core.events import ReplayEngine

        engine = ReplayEngine(self.session.chain)
        state = engine.replay_all()
        trace = engine.get_trace()

        # Agent stats
        agent_stats: Dict[str, Dict[str, Any]] = {}
        for step in trace:
            agent = step["agent"]
            if agent not in agent_stats:
                agent_stats[agent] = {"steps": 0, "duration_ms": 0}
            agent_stats[agent]["steps"] += 1
            agent_stats[agent]["duration_ms"] += step.get("duration_ms", 0)

        return {
            "chain_id": self.session.chain.id,
            "event_count": len(self.session.chain.events),
            "agent_stats": agent_stats,
            "tool_counts": self.tool_counts,
            "fact_count": len(state.facts),
            "error_count": len(state.errors),
            "total_duration_ms": state.metrics.get("total_duration_ms", 0),
        }

    def reset(self) -> None:
        """Reset the tracer for a new workflow."""
        global _tracer_instance, _tracer_file

        self.session = LCTLSession(
            chain_id=f"claude-code-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        self.agent_stack = []
        self.tool_counts = {}
        self._start_times = {}

        # Clear state file
        if _tracer_file and _tracer_file.exists():
            _tracer_file.unlink()

        _tracer_instance = None


def generate_hooks(output_dir: str = ".claude/hooks") -> Dict[str, str]:
    """Generate Claude Code hook scripts for LCTL tracing.

    Args:
        output_dir: Directory to write hook scripts

    Returns:
        Dict mapping hook name to file path
    """
    hooks_dir = Path(output_dir)
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hooks = {}

    # PreToolUse hook for Task tool
    pre_tool_hook = '''#!/bin/bash
# LCTL Tracing Hook - Pre Tool Use
# Records Task tool invocations as STEP_START events

if [ "$CLAUDE_TOOL_NAME" = "Task" ]; then
    python3 -c "
import os
import sys
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer

    tracer = LCTLClaudeCodeTracer.get_or_create()
    tracer.on_task_start(
        agent_type=os.environ.get('CLAUDE_TOOL_INPUT_subagent_type', 'unknown'),
        description=os.environ.get('CLAUDE_TOOL_INPUT_description', ''),
        prompt=os.environ.get('CLAUDE_TOOL_INPUT_prompt', ''),
        model=os.environ.get('CLAUDE_TOOL_INPUT_model'),
    )
except Exception as e:
    # Don't break Claude Code if tracing fails
    print(f'LCTL trace warning: {e}', file=sys.stderr)
"
fi
'''

    pre_hook_path = hooks_dir / "PreToolUse.sh"
    pre_hook_path.write_text(pre_tool_hook)
    pre_hook_path.chmod(0o755)
    hooks["PreToolUse"] = str(pre_hook_path)

    # PostToolUse hook for Task tool
    post_tool_hook = '''#!/bin/bash
# LCTL Tracing Hook - Post Tool Use
# Records Task tool completions as STEP_END events

if [ "$CLAUDE_TOOL_NAME" = "Task" ]; then
    python3 -c "
import os
import sys
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer

    tracer = LCTLClaudeCodeTracer.get_or_create()

    # Check for error in result
    result = os.environ.get('CLAUDE_TOOL_RESULT', '')
    success = 'error' not in result.lower()[:100]

    tracer.on_task_complete(
        agent_type=os.environ.get('CLAUDE_TOOL_INPUT_subagent_type', 'unknown'),
        description=os.environ.get('CLAUDE_TOOL_INPUT_description', ''),
        result=result[:2000],
        success=success,
    )
except Exception as e:
    print(f'LCTL trace warning: {e}', file=sys.stderr)
"
fi

# Also trace other tool calls
if [ "$CLAUDE_TOOL_NAME" != "Task" ]; then
    python3 -c "
import os
import sys
import json
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer

    tracer = LCTLClaudeCodeTracer.get_or_create()

    # Get tool input (simplified - full parsing would need more logic)
    tool_name = os.environ.get('CLAUDE_TOOL_NAME', 'unknown')

    # Skip high-frequency read-only tools to reduce noise
    if tool_name not in ('Read', 'Glob', 'Grep', 'LS'):
        tracer.on_tool_call(
            tool_name=tool_name,
            input_data={'raw': os.environ.get('CLAUDE_TOOL_INPUT', '')[:500]},
            output_data={'raw': os.environ.get('CLAUDE_TOOL_RESULT', '')[:500]},
        )
except Exception as e:
    pass  # Silent fail for non-Task tools
"
fi
'''

    post_hook_path = hooks_dir / "PostToolUse.sh"
    post_hook_path.write_text(post_tool_hook)
    post_hook_path.chmod(0o755)
    hooks["PostToolUse"] = str(post_hook_path)

    # Stop hook to export final trace
    stop_hook = '''#!/bin/bash
# LCTL Tracing Hook - Stop
# Exports final trace when Claude Code session ends

python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from lctl.integrations.claude_code import LCTLClaudeCodeTracer
    from pathlib import Path

    state_file = Path('.claude/traces/.lctl-state.json')
    if state_file.exists():
        tracer = LCTLClaudeCodeTracer.get_or_create()
        path = tracer.export()
        summary = tracer.get_summary()

        print(f'LCTL Trace exported: {path}')
        print(f'  Events: {summary[\"event_count\"]}')
        print(f'  Agents: {list(summary[\"agent_stats\"].keys())}')
        print(f'  Facts: {summary[\"fact_count\"]}')

        # Clean up state file
        state_file.unlink()
except Exception as e:
    print(f'LCTL export warning: {e}', file=sys.stderr)
"
'''

    stop_hook_path = hooks_dir / "Stop.sh"
    stop_hook_path.write_text(stop_hook)
    stop_hook_path.chmod(0o755)
    hooks["Stop"] = str(stop_hook_path)

    return hooks


def is_available() -> bool:
    """Check if Claude Code tracing is available.

    Returns:
        True (always available - no external dependencies)
    """
    return True


__all__ = [
    "LCTLClaudeCodeTracer",
    "generate_hooks",
    "is_available",
]
