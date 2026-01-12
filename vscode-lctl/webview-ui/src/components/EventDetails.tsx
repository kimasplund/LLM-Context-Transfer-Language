import { useStore } from '../store';
import { vscode } from '../App';
import CodeLinks from './CodeLinks';

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60000).toFixed(2)}m`;
}

export default function EventDetails() {
  const selectedEvent = useStore((s) => s.selectedEvent);
  const selectEvent = useStore((s) => s.selectEvent);

  if (!selectedEvent) {
    return (
      <div className="details-panel">
        <div className="details-header">
          <span>Event Details</span>
        </div>
        <div className="details-content" style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: 0.5
        }}>
          Select an event to view details
        </div>
      </div>
    );
  }

  const { seq, type, timestamp, agent, data } = selectedEvent;

  // Extract common fields for display
  const fields: Array<{ label: string; value: string; highlight?: boolean }> = [
    { label: 'Sequence', value: `#${seq}` },
    { label: 'Type', value: type },
    { label: 'Time', value: new Date(timestamp).toLocaleString() },
  ];

  if (agent) {
    fields.push({ label: 'Agent', value: agent });
  }

  // Type-specific fields
  if (type === 'step_start' && data) {
    if (data.intent) fields.push({ label: 'Intent', value: String(data.intent) });
    if (data.input_summary) fields.push({ label: 'Input', value: String(data.input_summary) });
  }

  if (type === 'step_end' && data) {
    if (data.outcome) fields.push({ label: 'Outcome', value: String(data.outcome), highlight: true });
    if (data.duration_ms) fields.push({ label: 'Duration', value: formatDuration(Number(data.duration_ms)) });
    if (data.tokens_in) fields.push({ label: 'Tokens In', value: Number(data.tokens_in).toLocaleString() });
    if (data.tokens_out) fields.push({ label: 'Tokens Out', value: Number(data.tokens_out).toLocaleString() });
  }

  if (type === 'tool_call' && data) {
    if (data.tool) fields.push({ label: 'Tool', value: String(data.tool), highlight: true });
    if (data.duration_ms) fields.push({ label: 'Duration', value: formatDuration(Number(data.duration_ms)) });
  }

  if ((type === 'fact_added' || type === 'fact_modified') && data) {
    if (data.id) fields.push({ label: 'Fact ID', value: String(data.id), highlight: true });
    if (data.confidence !== undefined) {
      const conf = Number(data.confidence);
      fields.push({
        label: 'Confidence',
        value: `${(conf * 100).toFixed(0)}%`,
        highlight: conf >= 0.8
      });
    }
    if (data.text) fields.push({ label: 'Text', value: String(data.text) });
  }

  if (type === 'error' && data) {
    if (data.error_type) fields.push({ label: 'Error Type', value: String(data.error_type), highlight: true });
    if (data.message) fields.push({ label: 'Message', value: String(data.message) });
  }

  // Check for file paths in data to enable navigation
  const filePath = (data?.file_path || data?.path || data?.source_file) as string | undefined;
  const lineNumber = (data?.line || data?.line_number) as number | undefined;

  const handleOpenFile = () => {
    if (filePath) {
      vscode.postMessage({
        type: 'openFile',
        path: String(filePath),
        line: lineNumber ? Number(lineNumber) : undefined
      });
    }
  };

  return (
    <div className="details-panel">
      <div className="details-header">
        <span>Event Details — #{seq}</span>
        <div style={{ display: 'flex', gap: 8 }}>
          {filePath && (
            <button className="button secondary" onClick={handleOpenFile}>
              📄 Open File
            </button>
          )}
          <button
            className="button secondary"
            onClick={() => selectEvent(null)}
          >
            ✕
          </button>
        </div>
      </div>
      <div className="details-content">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
          {fields.map((field) => (
            <div key={field.label} className="details-field">
              <div className="details-field-label">{field.label}</div>
              <div
                className="details-field-value"
                style={{
                  color: field.highlight ? 'var(--vscode-accent)' : undefined,
                  wordBreak: 'break-word'
                }}
              >
                {field.value}
              </div>
            </div>
          ))}
        </div>

        {/* Code links */}
        <CodeLinks event={selectedEvent} />

        {/* Raw data */}
        {data && Object.keys(data).length > 0 && (
          <div style={{ marginTop: 16 }}>
            <div className="details-field-label" style={{ marginBottom: 8 }}>Raw Data</div>
            <pre className="details-json">{JSON.stringify(data, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
