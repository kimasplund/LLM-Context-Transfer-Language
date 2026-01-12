import { useStore } from '../store';
import { vscode } from '../App';

export default function Header() {
  const chain = useStore((s) => s.chain);
  const isRecording = useStore((s) => s.isRecording);
  const isDiffMode = useStore((s) => s.isDiffMode);

  const handleExport = () => {
    vscode.postMessage({ type: 'exportHtml' });
  };

  const handleStats = () => {
    vscode.postMessage({ type: 'showStats' });
  };

  const handleCompare = () => {
    // Request to compare chains - this will trigger a file picker in the extension
    vscode.postMessage({ type: 'requestCompare' });
  };

  return (
    <header className="header">
      <div className="header-title">
        <span>🔍 LCTL Dashboard</span>
        {chain && <span style={{ opacity: 0.6 }}>— {chain.chain.id}</span>}
        {isDiffMode && <span style={{ color: 'var(--vscode-warning)' }}> (Diff Mode)</span>}
        {isRecording && (
          <span className="recording-badge">RECORDING</span>
        )}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        {!isDiffMode && (
          <button className="button secondary" onClick={handleCompare}>
            📊 Compare
          </button>
        )}
        <button className="button secondary" onClick={handleStats}>
          📈 Stats
        </button>
        <button className="button" onClick={handleExport}>
          📄 Export
        </button>
      </div>
    </header>
  );
}
