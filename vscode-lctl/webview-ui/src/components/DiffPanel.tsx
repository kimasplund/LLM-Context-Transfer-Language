import { useState, useMemo } from 'react';
import type { LctlChainFile, LctlEvent } from '../types';

interface DiffPanelProps {
  chain1: LctlChainFile;
  chain2: LctlChainFile | null;
  onClose: () => void;
}

interface DiffResult {
  type: 'added' | 'removed' | 'modified' | 'same';
  event1?: LctlEvent;
  event2?: LctlEvent;
  seq: number;
}

function computeDiff(chain1: LctlChainFile, chain2: LctlChainFile): DiffResult[] {
  const events1 = chain1.events || [];
  const events2 = chain2.events || [];
  const results: DiffResult[] = [];

  // Create lookup maps by seq
  const map1 = new Map(events1.map(e => [e.seq, e]));
  const map2 = new Map(events2.map(e => [e.seq, e]));

  // Get all unique seq numbers
  const allSeqs = new Set([...map1.keys(), ...map2.keys()]);
  const sortedSeqs = Array.from(allSeqs).sort((a, b) => a - b);

  for (const seq of sortedSeqs) {
    const e1 = map1.get(seq);
    const e2 = map2.get(seq);

    if (e1 && e2) {
      // Both exist - check if same or modified
      const same = JSON.stringify(e1) === JSON.stringify(e2);
      results.push({
        type: same ? 'same' : 'modified',
        event1: e1,
        event2: e2,
        seq
      });
    } else if (e1) {
      // Only in chain1 - removed in chain2
      results.push({ type: 'removed', event1: e1, seq });
    } else if (e2) {
      // Only in chain2 - added in chain2
      results.push({ type: 'added', event2: e2, seq });
    }
  }

  return results;
}

function DiffRow({ diff }: { diff: DiffResult }) {
  const [expanded, setExpanded] = useState(false);

  const getStyle = () => {
    switch (diff.type) {
      case 'added': return { background: 'rgba(78, 201, 176, 0.1)', borderLeft: '3px solid var(--vscode-success)' };
      case 'removed': return { background: 'rgba(241, 76, 76, 0.1)', borderLeft: '3px solid var(--vscode-error)' };
      case 'modified': return { background: 'rgba(204, 167, 0, 0.1)', borderLeft: '3px solid var(--vscode-warning)' };
      default: return { borderLeft: '3px solid var(--vscode-border)' };
    }
  };

  const getIcon = () => {
    switch (diff.type) {
      case 'added': return '+';
      case 'removed': return '-';
      case 'modified': return '~';
      default: return ' ';
    }
  };

  const event = diff.event1 || diff.event2;
  if (!event) return null;

  return (
    <div style={getStyle()}>
      <div
        style={{
          padding: '8px 12px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 12
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <span style={{
          width: 20,
          textAlign: 'center',
          fontWeight: 'bold',
          fontFamily: 'monospace'
        }}>
          {getIcon()}
        </span>
        <span style={{ fontFamily: 'monospace', opacity: 0.6 }}>{diff.seq}</span>
        <span>{event.type}</span>
        {event.agent && (
          <span style={{
            background: 'var(--vscode-badge-background)',
            color: 'var(--vscode-badge-foreground)',
            padding: '2px 6px',
            borderRadius: 3,
            fontSize: 10
          }}>
            {event.agent}
          </span>
        )}
      </div>

      {expanded && diff.type === 'modified' && (
        <div style={{
          padding: '0 12px 12px 44px',
          fontSize: 11
        }}>
          <div style={{ marginBottom: 8 }}>
            <div style={{ opacity: 0.6, marginBottom: 4 }}>Chain 1:</div>
            <pre style={{
              background: 'rgba(241, 76, 76, 0.1)',
              padding: 8,
              borderRadius: 4,
              overflow: 'auto',
              margin: 0
            }}>
              {JSON.stringify(diff.event1?.data, null, 2)}
            </pre>
          </div>
          <div>
            <div style={{ opacity: 0.6, marginBottom: 4 }}>Chain 2:</div>
            <pre style={{
              background: 'rgba(78, 201, 176, 0.1)',
              padding: 8,
              borderRadius: 4,
              overflow: 'auto',
              margin: 0
            }}>
              {JSON.stringify(diff.event2?.data, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {expanded && diff.type !== 'modified' && (
        <div style={{
          padding: '0 12px 12px 44px',
          fontSize: 11
        }}>
          <pre style={{
            background: 'var(--vscode-input-background)',
            padding: 8,
            borderRadius: 4,
            overflow: 'auto',
            margin: 0
          }}>
            {JSON.stringify(event.data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export default function DiffPanel({ chain1, chain2, onClose }: DiffPanelProps) {
  const [showSame, setShowSame] = useState(false);

  const diffs = useMemo(() => {
    if (!chain2) return [];
    return computeDiff(chain1, chain2);
  }, [chain1, chain2]);

  const filteredDiffs = useMemo(() => {
    if (showSame) return diffs;
    return diffs.filter(d => d.type !== 'same');
  }, [diffs, showSame]);

  const stats = useMemo(() => {
    return {
      added: diffs.filter(d => d.type === 'added').length,
      removed: diffs.filter(d => d.type === 'removed').length,
      modified: diffs.filter(d => d.type === 'modified').length,
      same: diffs.filter(d => d.type === 'same').length
    };
  }, [diffs]);

  if (!chain2) {
    return (
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 16
      }}>
        <div style={{ fontSize: 48, opacity: 0.3 }}>📊</div>
        <div>Select a second chain to compare</div>
        <button className="button secondary" onClick={onClose}>Cancel</button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--vscode-border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div>
          <div style={{ fontWeight: 600 }}>Chain Comparison</div>
          <div style={{ fontSize: 11, opacity: 0.7 }}>
            {chain1.chain.id} vs {chain2.chain.id}
          </div>
        </div>
        <button className="button secondary" onClick={onClose}>✕ Close</button>
      </div>

      {/* Stats */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--vscode-border)',
        display: 'flex',
        gap: 16,
        fontSize: 12
      }}>
        <span style={{ color: 'var(--vscode-success)' }}>+{stats.added} added</span>
        <span style={{ color: 'var(--vscode-error)' }}>-{stats.removed} removed</span>
        <span style={{ color: 'var(--vscode-warning)' }}>~{stats.modified} modified</span>
        <span style={{ opacity: 0.6 }}>{stats.same} unchanged</span>
        <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={showSame}
            onChange={(e) => setShowSame(e.target.checked)}
          />
          Show unchanged
        </label>
      </div>

      {/* Diff list */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {filteredDiffs.map((diff, i) => (
          <DiffRow key={i} diff={diff} />
        ))}
        {filteredDiffs.length === 0 && (
          <div style={{
            padding: 32,
            textAlign: 'center',
            opacity: 0.6
          }}>
            {diffs.length === 0
              ? 'Chains are identical'
              : 'No differences to show (try enabling "Show unchanged")'}
          </div>
        )}
      </div>
    </div>
  );
}
