import { useStore } from '../store';

export default function StatsGrid() {
  const stats = useStore((s) => s.stats);

  if (!stats) return null;

  const items = [
    { label: 'Events', value: stats.eventCount.toLocaleString(), className: '' },
    { label: 'Agents', value: stats.agents.length.toString(), className: '' },
    { label: 'Duration', value: `${(stats.totalDurationMs / 1000).toFixed(1)}s`, className: '' },
    { label: 'Tokens In', value: stats.tokensIn.toLocaleString(), className: '' },
    { label: 'Tokens Out', value: stats.tokensOut.toLocaleString(), className: '' },
    { label: 'Est. Cost', value: `$${stats.estimatedCost.toFixed(3)}`, className: '' },
    { label: 'Errors', value: stats.errorCount.toString(), className: stats.errorCount > 0 ? 'error' : '' },
    { label: 'Tools', value: stats.tools.length.toString(), className: '' },
  ];

  return (
    <div className="stats-grid">
      {items.map((item) => (
        <div key={item.label} className="stat-card">
          <div className={`stat-value ${item.className}`}>{item.value}</div>
          <div className="stat-label">{item.label}</div>
        </div>
      ))}
    </div>
  );
}
