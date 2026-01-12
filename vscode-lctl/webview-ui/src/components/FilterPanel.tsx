import { useStore } from '../store';
import type { EventType } from '../types';

const EVENT_TYPES: Array<{ type: EventType; label: string; icon: string }> = [
  { type: 'step_start', label: 'Step Start', icon: '▶️' },
  { type: 'step_end', label: 'Step End', icon: '⏹️' },
  { type: 'tool_call', label: 'Tool Call', icon: '🔧' },
  { type: 'fact_added', label: 'Fact Added', icon: '💡' },
  { type: 'fact_modified', label: 'Fact Modified', icon: '✏️' },
  { type: 'error', label: 'Error', icon: '❌' },
  { type: 'checkpoint', label: 'Checkpoint', icon: '📍' },
];

export default function FilterPanel() {
  const filter = useStore((s) => s.filter);
  const stats = useStore((s) => s.stats);
  const setFilter = useStore((s) => s.setFilter);

  const toggleEventType = (type: EventType) => {
    const types = filter.eventTypes.includes(type)
      ? filter.eventTypes.filter((t) => t !== type)
      : [...filter.eventTypes, type];
    setFilter({ eventTypes: types });
  };

  const toggleAgent = (agent: string) => {
    const agents = filter.agents.includes(agent)
      ? filter.agents.filter((a) => a !== agent)
      : [...filter.agents, agent];
    setFilter({ agents });
  };

  const clearFilters = () => {
    setFilter({
      eventTypes: [],
      agents: [],
      searchQuery: '',
      timeRange: null
    });
  };

  const hasActiveFilters = filter.eventTypes.length > 0 ||
                           filter.agents.length > 0 ||
                           filter.searchQuery !== '';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Search */}
      <div className="filter-section">
        <input
          type="text"
          className="search-input"
          placeholder="Search events..."
          value={filter.searchQuery}
          onChange={(e) => setFilter({ searchQuery: e.target.value })}
        />
      </div>

      {/* Event Type Filter */}
      <div className="filter-section">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 11, textTransform: 'uppercase', opacity: 0.6 }}>Event Types</span>
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              style={{
                fontSize: 10,
                background: 'none',
                border: 'none',
                color: 'var(--vscode-accent)',
                cursor: 'pointer'
              }}
            >
              Clear all
            </button>
          )}
        </div>
        <div className="filter-chips">
          {EVENT_TYPES.map(({ type, label, icon }) => (
            <button
              key={type}
              className={`filter-chip ${filter.eventTypes.includes(type) ? 'active' : ''}`}
              onClick={() => toggleEventType(type)}
            >
              {icon} {label}
            </button>
          ))}
        </div>
      </div>

      {/* Agent Filter */}
      {stats && stats.agents.length > 0 && (
        <div className="filter-section">
          <span style={{ fontSize: 11, textTransform: 'uppercase', opacity: 0.6 }}>Agents</span>
          <div className="filter-chips">
            {stats.agents.map((agent) => (
              <button
                key={agent}
                className={`filter-chip ${filter.agents.includes(agent) ? 'active' : ''}`}
                onClick={() => toggleAgent(agent)}
              >
                {agent}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Facts Panel */}
      {stats && stats.facts.length > 0 && (
        <div className="filter-section" style={{ flex: 1, overflow: 'auto' }}>
          <span style={{ fontSize: 11, textTransform: 'uppercase', opacity: 0.6 }}>
            Facts ({stats.facts.length})
          </span>
          <div style={{ marginTop: 8 }}>
            {stats.facts.map((fact) => (
              <div
                key={fact.id}
                style={{
                  padding: '8px',
                  marginBottom: '6px',
                  background: 'var(--vscode-input-background)',
                  borderRadius: '4px',
                  fontSize: '11px'
                }}
              >
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: '4px'
                }}>
                  <span style={{ fontWeight: 600, color: 'var(--color-fact)' }}>{fact.id}</span>
                  <span style={{
                    color: fact.confidence >= 0.8 ? 'var(--vscode-success)' :
                           fact.confidence >= 0.5 ? 'var(--vscode-warning)' : 'var(--vscode-error)'
                  }}>
                    {(fact.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ opacity: 0.8, wordBreak: 'break-word' }}>
                  {fact.text.length > 100 ? fact.text.slice(0, 100) + '...' : fact.text}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
