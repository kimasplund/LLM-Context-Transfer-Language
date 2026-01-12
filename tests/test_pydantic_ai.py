
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from lctl.core.session import LCTLSession
from lctl.integrations.pydantic_ai import trace_agent, TracedAgent, LCTLPydanticAITracer

# Mock PydanticAI classes if not available
try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.usage import RunUsage as Usage
except ImportError:
    pytest.skip("pydantic-ai not installed", allow_module_level=True)

@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=Agent)
    agent.name = "test_agent"
    # Mock run method to be async
    agent.run = AsyncMock()
    return agent

@pytest.fixture
def mock_result():
    result = MagicMock(spec=AgentRunResult)
    result.data = "Test output"
    
    usage = MagicMock(spec=Usage)
    usage.request_tokens = 10
    usage.response_tokens = 20
    
    # In PydanticAI v1.x, usage is a method or property often
    # Let's mock usage() method
    result.usage = MagicMock(return_value=usage)
    return result

@pytest.mark.asyncio
async def test_trace_agent_run_success(mock_agent, mock_result):
    mock_agent.run.return_value = mock_result
    
    traced = trace_agent(mock_agent, chain_id="test-chain")
    
    # Mock session methods
    traced.tracer.session.step_start = MagicMock()
    traced.tracer.session.step_end = MagicMock()
    
    result = await traced.run("test input")
    
    assert result == mock_result
    traced.tracer.session.step_start.assert_called_once()
    traced.tracer.session.step_end.assert_called_once_with(
        outcome="success", 
        output_summary="Test output",
        tokens_in=10,
        tokens_out=20
    )

@pytest.mark.asyncio
async def test_trace_agent_run_error(mock_agent):
    mock_agent.run.side_effect = ValueError("Test error")
    
    traced = trace_agent(mock_agent, chain_id="test-chain")
    
    traced.tracer.session.step_start = MagicMock()
    traced.tracer.session.step_end = MagicMock()
    traced.tracer.session.error = MagicMock()
    
    with pytest.raises(ValueError):
        await traced.run("test input")
        
    traced.tracer.session.step_start.assert_called_once()
    traced.tracer.session.error.assert_called_once()
    traced.tracer.session.error.assert_called_once()
    traced.tracer.session.step_end.assert_called_with(outcome="error")

@pytest.mark.asyncio
async def test_trace_agent_tool_call(mock_agent, mock_result):
    # Setup mock tools
    mock_tool_func = AsyncMock(return_value="tool_result")
    mock_tool = MagicMock()
    mock_tool.function = mock_tool_func
    
    # Mock the internal structure
    mock_agent._function_toolset = MagicMock()
    mock_agent._function_toolset.tools = {"test_tool": mock_tool}
    
    mock_agent.run.return_value = mock_result
    
    traced = trace_agent(mock_agent, chain_id="test-chain")
    
    # Mock session
    traced.tracer.session.tool_call = MagicMock()
    
    # We need to trigger the tool execution. 
    # Since we monkeypatched the tool function in __init__, we can call it directly 
    # to simulate the agent calling it.
    await mock_tool.function("arg1", kwarg1="value")
    
    traced.tracer.session.tool_call.assert_called_once()
    # Check arguments
    call_args = traced.tracer.session.tool_call.call_args[1]
    assert call_args["tool"] == "test_tool"
    assert "tool_result" in call_args["output_data"]
