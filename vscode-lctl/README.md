# LCTL Time-Travel Debugger

The **LCTL Time-Travel Debugger** for VS Code brings the power of the LLM Context Transfer Language (LCTL) directly into your editor. It provides a visual interface to explore, debug, and replay agent execution chains.

## Features

### 🔍 Explorer Chain View
- Automatically detects `.lctl.json` files in your workspace.
- Organizing chains by ID and Snapshots.
- Quick navigation through your agent's history.

### 📊 Visual Dashboard
A comprehensive visual interface for your agent traces:
- **Timeline**: See the sequence of events (Step Start, Tool Call, LLM Trace, Error).
- **Facts**: View the evolving shared memory (facts) and their confidence levels.
- **Detailed Events**: Inspect raw LLM token usage, tool inputs, and step intents.
- **Swimlanes**: Visualize interactions between multiple agents.

### ⏪ Time-Travel Replay
- **Real Execution**: Integrated directly with the `lctl` CLI.
- **Replay**: Click the "Replay" button to spawn a terminal and replay the chain up to any point.
- **Debug**: Step through your agent's logic with the exact context from the past.

## Requirements

The extension relies on the `lctl` Python package.

```bash
pip install git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git
```

Ensure `lctl` is available in your system path or active virtual environment.

## Usage

1.  **Open Workspace**: Open a folder containing `.lctl.json` trace files.
2.  **LCTL Explorer**: Navigate to the "LCTL Explorer" view in the Side Bar.
3.  **Load Chain**: Click on a chain or snapshot file manifest to open the visual dashboard.
4.  **Inspect**: Click on timeline events to verify data flow.
5.  **Replay**: Click the "Replay Chain" button in the dashboard header to start a replay session in the terminal.

## Troubleshooting

- **"No LCTL chains found"**: Ensure your trace files end with `.lctl.json` and are in the root or subdirectories of your workspace.
- **"Could not find chain file for replay"**: Ensure you have loaded the chain in the dashboard first.
- **Feature Missing?**: This extension is designed for LCTL v4.0+. Ensure your trace files are compatible.
