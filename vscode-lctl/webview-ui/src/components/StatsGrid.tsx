import { useMemo } from 'react';
import { useStore } from '../store';

interface StatItem {
  label: string;
  value: string;
  subValue?: string;
  icon: string;
  color: string;
  percentage?: number;
  trend?: 'up' | 'down' | 'neutral';
}

function ProgressBar({ value, color, animated = false }: { value: number; color: string; animated?: boolean }) {
  return (
    <div className="stat-progress">
      <div
        className={`stat-progress-fill ${animated ? 'animated' : ''}`}
        style={{
          width: `${Math.min(100, Math.max(0, value))}%`,
          background: color
        }}
      />
    </div>
  );
}

function TrendIndicator({ trend }: { trend: 'up' | 'down' | 'neutral' }) {
  const icons = { up: '↑', down: '↓', neutral: '→' };
  const colors = {
    up: 'var(--vscode-success)',
    down: 'var(--vscode-error)',
    neutral: 'var(--vscode-foreground)'
  };
  return (
    <span className="stat-trend" style={{ color: colors[trend] }}>
      {icons[trend]}
    </span>
  );
}

export default function StatsGrid() {
  const stats = useStore((s) => s.stats);
  const chain = useStore((s) => s.chain);

  const items: StatItem[] = useMemo(() => {
    if (!stats) return [];

    // Calculate percentages for visual indicators
    const maxTokens = Math.max(stats.tokensIn, stats.tokensOut, 1);
    const tokenInPct = (stats.tokensIn / maxTokens) * 100;
    const tokenOutPct = (stats.tokensOut / maxTokens) * 100;

    // Error rate
    const errorRate = stats.eventCount > 0 ? (stats.errorCount / stats.eventCount) * 100 : 0;

    // Tool usage intensity
    const toolIntensity = stats.eventCount > 0 ? (stats.tools.length / Math.sqrt(stats.eventCount)) * 30 : 0;

    return [
      {
        label: 'Events',
        value: stats.eventCount.toLocaleString(),
        subValue: `${stats.agents.length} agents`,
        icon: '📊',
        color: 'var(--vscode-accent)',
        percentage: 100
      },
      {
        label: 'Duration',
        value: formatDuration(stats.totalDurationMs),
        subValue: `${(stats.totalDurationMs / Math.max(stats.eventCount, 1)).toFixed(0)}ms/event`,
        icon: '⏱️',
        color: 'var(--color-step-end)',
        percentage: Math.min(100, stats.totalDurationMs / 1000) // Scale: 1s = 100%
      },
      {
        label: 'Tokens In',
        value: formatNumber(stats.tokensIn),
        subValue: `${((stats.tokensIn / Math.max(stats.tokensIn + stats.tokensOut, 1)) * 100).toFixed(0)}% of total`,
        icon: '📥',
        color: 'var(--color-step-start)',
        percentage: tokenInPct
      },
      {
        label: 'Tokens Out',
        value: formatNumber(stats.tokensOut),
        subValue: `${((stats.tokensOut / Math.max(stats.tokensIn + stats.tokensOut, 1)) * 100).toFixed(0)}% of total`,
        icon: '📤',
        color: 'var(--color-fact)',
        percentage: tokenOutPct
      },
      {
        label: 'Est. Cost',
        value: `$${stats.estimatedCost.toFixed(3)}`,
        subValue: `$${(stats.estimatedCost / Math.max(stats.eventCount, 1) * 1000).toFixed(2)}/k events`,
        icon: '💰',
        color: 'var(--vscode-warning)',
        percentage: Math.min(100, stats.estimatedCost * 100) // $1 = 100%
      },
      {
        label: 'Errors',
        value: stats.errorCount.toString(),
        subValue: errorRate > 0 ? `${errorRate.toFixed(1)}% error rate` : 'No errors',
        icon: stats.errorCount > 0 ? '❌' : '✅',
        color: stats.errorCount > 0 ? 'var(--vscode-error)' : 'var(--vscode-success)',
        percentage: errorRate,
        trend: stats.errorCount > 0 ? 'down' : 'neutral'
      },
      {
        label: 'Tools',
        value: stats.tools.length.toString(),
        subValue: stats.tools.slice(0, 3).join(', ') + (stats.tools.length > 3 ? '...' : ''),
        icon: '🔧',
        color: 'var(--color-tool-call)',
        percentage: Math.min(100, toolIntensity)
      },
      {
        label: 'Facts',
        value: countFacts(chain).toString(),
        subValue: 'Registered facts',
        icon: '💡',
        color: 'var(--color-fact)',
        percentage: Math.min(100, countFacts(chain) * 10)
      },
    ];
  }, [stats, chain]);

  if (!stats) return null;

  return (
    <div className="stats-grid-enhanced">
      {items.map((item, index) => (
        <div
          key={item.label}
          className="stat-card-enhanced"
          style={{
            animationDelay: `${index * 50}ms`,
            '--accent-color': item.color
          } as React.CSSProperties}
        >
          <div className="stat-card-header">
            <span className="stat-icon">{item.icon}</span>
            <span className="stat-label-enhanced">{item.label}</span>
            {item.trend && <TrendIndicator trend={item.trend} />}
          </div>
          <div className="stat-value-enhanced" style={{ color: item.color }}>
            {item.value}
          </div>
          {item.subValue && (
            <div className="stat-subvalue">{item.subValue}</div>
          )}
          {item.percentage !== undefined && (
            <ProgressBar
              value={item.percentage}
              color={item.color}
              animated={item.label === 'Duration'}
            />
          )}
        </div>
      ))}
    </div>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`;
  return `${(ms / 3600000).toFixed(1)}h`;
}

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toLocaleString();
}

function countFacts(chain: { events?: Array<{ type: string }> } | null): number {
  if (!chain?.events) return 0;
  return chain.events.filter((e: { type: string }) =>
    e.type === 'fact_added' || e.type === 'fact_modified'
  ).length;
}
