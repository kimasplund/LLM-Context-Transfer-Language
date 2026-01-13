# LCTL Extension User Manual

This manual provides a detailed guide on using the **LCTL Time-Travel Debugger** extension for Visual Studio Code.

## Table of Contents
1.  [Getting Started](#getting-started)
2.  [Workflow Overview](#workflow-overview)
3.  [The Dashboard](#the-dashboard)
4.  [Replay Functionality](#replay-functionality)
5.  [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites
- VS Code 1.85 or higher.
- Python 3.9+ installed.
- `lctl` package installed:
  ```bash
  pip install git+https://github.com/kimasplund/LLM-Context-Trace-Library.git
  ```

### Installation
1.  Install the `.vsix` file provided in the release.
2.  Or build from source:
    ```bash
    cd vscode-lctl
    npm install
    npm run compile
    code --install-extension lctl-debugger-0.1.0.vsix
    ```

## Workflow Overview

The typical workflow involves generating a trace with your agent, then analyzing it in VS Code.

1.  **Run Agent**: Execute your PydanticAI/LangChain/CrewAI agent with LCTL tracing enabled.
    ```python
    # ... inside your python code ...
    chain.save("my-trace.lctl.json")
    ```
2.  **Open in VS Code**: Open the folder containing `my-trace.lctl.json`.
3.  **Analyze**: Use the LCTL Explorer to open the trace.
4.  **Debug**: Replay specific sections to diagnose issues.

## The Dashboard

The Dashboard is your command center for trace analysis.

### 1. Header & Metadata
Top section displays:
- **Chain ID**: Unique identifier for the trace.
- **Version**: LCTL schema version.
- **Timestamp**: When the trace was created.

### 2. Timeline
A chronological visual representation of execution.
- **Steps**: Represented by blocks. Green = Success, Red = Error.
- **Tool Calls**: Blue icons indicating external interactions.
- **Checkpoints**: Purple markers for saved states.

### 3. Detailed Events
Located below the timeline, this list provides granular data:
- **LLM Traces**: Shows model name (e.g., `gpt-4`) and token usage (Input/Output).
- **Tool Inputs**: truncated view of the data sent to tools.
- **Step Intents**: The goal of the specific agent step.

## Replay Functionality

The killer feature of LCTL is **Time-Travel**.

1.  Open a chain in the Dashboard.
2.  Click the **Replay Chain** button in the top right.
3.  A new terminal **"LCTL Replay"** will open.
4.  The extension automatically executes:
    ```bash
    lctl replay "/absolute/path/to/your/chain.lctl.json"
    ```
5.  Interact with the CLI to step through events, inspect state, or rewind execution.

## Troubleshooting

### "Command 'lctl' not found"
If the Replay execution fails in the terminal:
- Ensure your Python environment is active.
- Try running `pip install lctl` in that same terminal.

### Empty Dashboard
- Check if your `.lctl.json` file is valid JSON.
- Verify it has the required fields: `chain_id`, `version`, and `events`.
