import { useMemo, useState } from 'react';
import { useStore, useTimelineTracks, useVisibleEvents } from '../store';
import type { LctlEvent } from '../types';

const EVENT_ICONS: Record<string, string> = {
  step_start: '▶',
  step_end: '⏹',
  tool_call: '🔧',
  fact_added: '💡',
  fact_modified: '✏',
  error: '❌',
  checkpoint: '📍',
};

function getEventLabel(event: LctlEvent): string {
  if (event.type === 'tool_call' && event.data?.tool) {
    return event.data.tool as string;
  }
  if (event.type === 'step_start' && event.data?.intent) {
    return event.data.intent as string;
  }
  if ((event.type === 'fact_added' || event.type === 'fact_modified') && event.data?.id) {
    return event.data.id as string;
  }
  if (event.type === 'error' && event.data?.error_type) {
    return event.data.error_type as string;
  }
  return event.type;
}

function EventMarker({ event }: { event: LctlEvent }) {
  const selectedEvent = useStore((s) => s.selectedEvent);
  const selectEvent = useStore((s) => s.selectEvent);
  const replay = useStore((s) => s.replay);
  const isSelected = selectedEvent?.seq === event.seq;
  const isPast = event.seq <= replay.currentSeq;

  return (
    <div
      className={`event-marker ${event.type} ${isSelected ? 'selected' : ''}`}
      style={{ opacity: isPast ? 1 : 0.3 }}
      onClick={() => selectEvent(event)}
      title={`[${event.seq}] ${event.type}${event.agent ? ` (${event.agent})` : ''}`}
    >
      <span>{EVENT_ICONS[event.type] || '•'}</span>
      <span>{getEventLabel(event)}</span>
    </div>
  );
}

function TrackView({ agent, events }: { agent: string; events: LctlEvent[] }) {
  const agentInitial = agent === '_global' ? '🌐' : agent.charAt(0).toUpperCase();
  const displayName = agent === '_global' ? 'Global' : agent;

  return (
    <div className="timeline-track">
      <div className="track-header">
        <div className="agent-icon">{agentInitial}</div>
        <span>{displayName}</span>
        <span style={{ opacity: 0.5, fontSize: 10 }}>({events.length})</span>
      </div>
      <div className="track-events">
        {events.map((event) => (
          <EventMarker key={event.seq} event={event} />
        ))}
      </div>
    </div>
  );
}

function SwimLaneView() {
  const chain = useStore((s) => s.chain);
  const visibleEvents = useVisibleEvents();
  const replay = useStore((s) => s.replay);
  const selectedEvent = useStore((s) => s.selectedEvent);
  const selectEvent = useStore((s) => s.selectEvent);

  // Group events by agent
  const tracks = useMemo(() => {
    const trackMap = new Map<string, LctlEvent[]>();
    for (const event of visibleEvents) {
      const agent = event.agent || '_global';
      if (!trackMap.has(agent)) {
        trackMap.set(agent, []);
      }
      trackMap.get(agent)!.push(event);
    }
    return Array.from(trackMap.entries())
      .map(([agent, events]) => ({ agent, events }))
      .sort((a, b) => {
        if (a.agent === '_global') return -1;
        if (b.agent === '_global') return 1;
        return a.agent.localeCompare(b.agent);
      });
  }, [visibleEvents]);

  if (!chain?.events?.length) {
    return (
      <div className="empty-state" style={{ height: 200 }}>
        <div>No events to display</div>
      </div>
    );
  }

  // Calculate timeline dimensions
  const minSeq = 0;
  const maxSeq = chain.events.length;
  const width = Math.max(800, maxSeq * 30);

  const getPosition = (seq: number) => {
    return ((seq - minSeq) / (maxSeq - minSeq || 1)) * 100;
  };

  return (
    <div className="swim-lanes" style={{ minWidth: width }}>
      {/* Time axis */}
      <div style={{
        height: 24,
        borderBottom: '1px solid var(--vscode-border)',
        marginBottom: 8,
        position: 'relative',
        marginLeft: 100
      }}>
        {[0, 25, 50, 75, 100].map((pct) => {
          const seq = Math.round((pct / 100) * maxSeq);
          return (
            <span
              key={pct}
              style={{
                position: 'absolute',
                left: `${pct}%`,
                transform: 'translateX(-50%)',
                fontSize: 10,
                opacity: 0.5
              }}
            >
              {seq}
            </span>
          );
        })}
      </div>

      {/* Playhead */}
      <div
        style={{
          position: 'absolute',
          left: `calc(100px + ${getPosition(replay.currentSeq)}%)`,
          top: 24,
          bottom: 0,
          width: 2,
          background: 'var(--vscode-accent)',
          opacity: 0.6,
          pointerEvents: 'none',
          zIndex: 5
        }}
      />

      {/* Swim lanes */}
      {tracks.map(({ agent, events }) => (
        <div key={agent} className="swim-lane">
          <div className="swim-lane-label">{agent === '_global' ? 'Global' : agent}</div>
          <div className="swim-lane-track" style={{ width }}>
            {events.map((event) => {
              const left = getPosition(event.seq);
              const isSelected = selectedEvent?.seq === event.seq;

              return (
                <div
                  key={event.seq}
                  className={`swim-lane-event ${event.type}`}
                  style={{
                    left: `${left}%`,
                    background: isSelected ? 'var(--vscode-accent)' : undefined,
                    borderColor: isSelected ? 'var(--vscode-accent)' : undefined
                  }}
                  onClick={() => selectEvent(event)}
                  title={`[${event.seq}] ${event.type}`}
                >
                  {EVENT_ICONS[event.type] || '•'}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Timeline() {
  const tracks = useTimelineTracks();
  const [viewMode, setViewMode] = useState<'track' | 'swimlane'>('track');

  return (
    <div className="timeline-container">
      <div className="timeline-header">
        <span style={{ fontWeight: 600 }}>Timeline</span>
        <div className="timeline-controls">
          <button
            className={`filter-chip ${viewMode === 'track' ? 'active' : ''}`}
            onClick={() => setViewMode('track')}
          >
            Grouped
          </button>
          <button
            className={`filter-chip ${viewMode === 'swimlane' ? 'active' : ''}`}
            onClick={() => setViewMode('swimlane')}
          >
            Swim Lanes
          </button>
        </div>
      </div>

      {viewMode === 'track' ? (
        tracks.length > 0 ? (
          tracks.map(({ agent, events }) => (
            <TrackView key={agent} agent={agent} events={events} />
          ))
        ) : (
          <div className="empty-state" style={{ height: 150 }}>
            <div>No events match current filters</div>
          </div>
        )
      ) : (
        <div style={{ overflow: 'auto' }}>
          <SwimLaneView />
        </div>
      )}
    </div>
  );
}
