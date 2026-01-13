# LCTL Time-Travel Debugger

The **LCTL Time-Travel Debugger** for VS Code brings the power of the LLM Context Trace Library (LCTL) directly into your editor. It provides a visual interface to explore, debug, and replay agent execution chains.

## Features

### LCTL Explorer
- Automatically detects `.lctl.json` files in your workspace
- **Live Updates**: Auto-refresh when chain files change
- **Status Badges**: Event counts, cost estimates, and error indicators
- **Recording Indicator**: Shows when LCTL tracing is active

### Interactive Dashboard
- **Timeline Visualization**: See events across agents in grouped or swim-lane view
- **Replay Controls**: Step through events with play/pause, forward/backward
- **Event Details**: Detailed view of each event with raw JSON data
- **Search & Filter**: Find events by type, agent, or content
- **Statistics Grid**: Event counts, token usage, duration, cost estimates

### Code Navigation
- **File Links**: Click to open referenced files at specific lines
- **Git Integration**: See git commit information from tool calls

### Chain Analysis
- **Diff/Compare**: Visual comparison between two chains
- **Export HTML**: Generate standalone HTML reports
- **Export Markdown**: Generate Markdown reports

## Requirements

The extension relies on the `lctl` Python package.

```bash
pip install git+https://github.com/kimasplund/LLM-Context-Transfer-Language.git
```

Ensure `lctl` is available in your system path or active virtual environment.

## Usage

### Getting Started

1. Initialize LCTL tracing in your project:
   ```bash
   lctl claude init --chain-id my-project
   ```

2. Open the LCTL Explorer in the VS Code sidebar

3. Click on a chain file to open the dashboard

### Commands

| Command | Description |
|---------|-------------|
| `LCTL: Open Dashboard` | Open the main LCTL dashboard |
| `LCTL: Load Chain` | Load a specific chain file |
| `LCTL: Replay Chain` | Replay a chain in terminal |
| `LCTL: Export HTML Report` | Export chain as standalone HTML |
| `LCTL: Export Markdown Report` | Export chain as Markdown |
| `LCTL: Show Statistics` | View chain statistics in terminal |
| `LCTL: Compare Chains` | Compare two chain files |
| `LCTL: Refresh` | Refresh the explorer view |

### Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `lctl.autoRefresh` | Auto-refresh explorer when files change | `true` |
| `lctl.maxEventsInTree` | Maximum events to show in tree view | `50` |
| `lctl.defaultPricing` | Model for cost estimation | `claude-sonnet-4` |

## Troubleshooting

- **"No LCTL chains found"**: Ensure your trace files end with `.lctl.json` and are in the root or subdirectories of your workspace.
- **"Could not find chain file for replay"**: Ensure you have loaded the chain in the dashboard first.
- **Feature Missing?**: This extension is designed for LCTL v4.0+. Ensure your trace files are compatible.

## License

AGPL-3.0-only

## Links

- [LCTL Documentation](https://github.com/kimasplund/LLM-Context-Transfer-Language)
- [Report Issues](https://github.com/kimasplund/LLM-Context-Transfer-Language/issues)
