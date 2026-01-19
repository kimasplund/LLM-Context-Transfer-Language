import { useMemo, useRef, useCallback } from 'react';
import { useStore } from '../store';

const EVENT_COLORS: Record<string, string> = {
  step_start: 'var(--color-step-start)',
  step_end: 'var(--color-step-end)',
  tool_call: 'var(--color-tool-call)',
  fact_added: 'var(--color-fact)',
  fact_modified: 'var(--color-fact)',
  error: 'var(--vscode-error)',
  checkpoint: 'var(--color-checkpoint)',
};

interface MinimapProps {
  height?: number;
}

export default function Minimap({ height = 60 }: MinimapProps) {
  const chain = useStore((s) => s.chain);
  const replay = useStore((s) => s.replay);
  const setReplayState = useStore((s) => s.setReplayState);
  const selectEvent = useStore((s) => s.selectEvent);
  const containerRef = useRef<HTMLDivElement>(null);

  const events = chain?.events || [];
  const totalEvents = events.length;

  // Group events into buckets for visualization
  const buckets = useMemo(() => {
    if (totalEvents === 0) return [];

    const bucketCount = 100; // Number of segments in minimap
    const bucketSize = Math.ceil(totalEvents / bucketCount);
    const result: Array<{
      startSeq: number;
      endSeq: number;
      counts: Record<string, number>;
      total: number;
    }> = [];

    for (let i = 0; i < bucketCount; i++) {
      const startSeq = i * bucketSize;
      const endSeq = Math.min((i + 1) * bucketSize, totalEvents);
      const bucketEvents = events.slice(startSeq, endSeq);

      const counts: Record<string, number> = {};
      for (const event of bucketEvents) {
        counts[event.type] = (counts[event.type] || 0) + 1;
      }

      result.push({
        startSeq,
        endSeq,
        counts,
        total: bucketEvents.length
      });
    }

    return result;
  }, [events, totalEvents]);

  // Find max bucket count for scaling
  const maxBucketTotal = useMemo(() => {
    return Math.max(1, ...buckets.map(b => b.total));
  }, [buckets]);

  // Handle click on minimap to jump to position
  const handleClick = useCallback((e: React.MouseEvent) => {
    if (!containerRef.current || totalEvents === 0) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = x / rect.width;
    const targetSeq = Math.round(percentage * totalEvents);

    setReplayState({ currentSeq: Math.max(0, Math.min(targetSeq, totalEvents)) });

    // Select the event at this position
    const event = events[targetSeq - 1];
    if (event) {
      selectEvent(event);
    }
  }, [totalEvents, events, setReplayState, selectEvent]);

  // Handle drag on minimap
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!containerRef.current || totalEvents === 0) return;

    e.preventDefault();

    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(moveEvent.clientX - rect.left, rect.width));
      const percentage = x / rect.width;
      const targetSeq = Math.round(percentage * totalEvents);

      setReplayState({ currentSeq: Math.max(0, Math.min(targetSeq, totalEvents)) });
    };

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    // Also handle initial click position
    handleClick(e);
  }, [totalEvents, setReplayState, handleClick]);

  if (totalEvents === 0) {
    return null;
  }

  const playheadPosition = (replay.currentSeq / totalEvents) * 100;

  return (
    <div className="minimap-container" style={{ height }}>
      <div className="minimap-label">
        <span className="minimap-title">Timeline Overview</span>
        <span className="minimap-position">
          {replay.currentSeq} / {totalEvents}
        </span>
      </div>
      <div
        ref={containerRef}
        className="minimap-track"
        onClick={handleClick}
        onMouseDown={handleMouseDown}
        role="slider"
        aria-label="Timeline minimap"
        aria-valuemin={0}
        aria-valuemax={totalEvents}
        aria-valuenow={replay.currentSeq}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft') {
            e.preventDefault();
            setReplayState({ currentSeq: Math.max(0, replay.currentSeq - 1) });
          } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            setReplayState({ currentSeq: Math.min(totalEvents, replay.currentSeq + 1) });
          }
        }}
      >
        {/* Event density bars */}
        <div className="minimap-bars">
          {buckets.map((bucket, index) => {
            const barHeight = (bucket.total / maxBucketTotal) * 100;
            const hasError = bucket.counts['error'] > 0;

            return (
              <div
                key={index}
                className={`minimap-bar ${hasError ? 'has-error' : ''}`}
                style={{ height: `${barHeight}%` }}
                title={`Events ${bucket.startSeq}-${bucket.endSeq}: ${bucket.total}`}
              >
                {/* Stacked colors by event type */}
                {Object.entries(bucket.counts).map(([type, count]) => (
                  <div
                    key={type}
                    className="minimap-bar-segment"
                    style={{
                      height: `${(count / bucket.total) * 100}%`,
                      background: EVENT_COLORS[type] || 'var(--vscode-foreground)',
                    }}
                  />
                ))}
              </div>
            );
          })}
        </div>

        {/* Playhead indicator */}
        <div
          className="minimap-playhead"
          style={{ left: `${playheadPosition}%` }}
        />

        {/* Hover zone for better interaction */}
        <div className="minimap-hover-zone" />
      </div>

      {/* Legend */}
      <div className="minimap-legend">
        {Object.entries(EVENT_COLORS).slice(0, 5).map(([type, color]) => (
          <span key={type} className="minimap-legend-item">
            <span className="minimap-legend-color" style={{ background: color }} />
            <span className="minimap-legend-label">{type.replace('_', ' ')}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
