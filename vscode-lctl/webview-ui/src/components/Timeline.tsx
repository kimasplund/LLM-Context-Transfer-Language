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

const EVENT_WIDTH = 28; // Minimum width of an event marker in pixels

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

// Collision detection: assign row to each event to prevent overlap
interface PositionedEvent {
  event: LctlEvent;
  left: number; // percentage
  row: number;  // vertical row for collision avoidance
}

function layoutEventsWithCollisionDetection(
  events: LctlEvent[],
  getPosition: (seq: number) => number,
  trackWidth: number,
  zoomLevel: number
): PositionedEvent[] {
  const positioned: PositionedEvent[] = [];
  const rows: number[] = []; // End position of each row

  const eventWidthPercent = (EVENT_WIDTH / (trackWidth * zoomLevel)) * 100;

  for (const event of events) {
    const left = getPosition(event.seq);

    // Find first row where this event doesn't overlap
    let row = 0;
    while (row < rows.length && rows[row] > left) {
      row++;
    }

    // Set end position for this row
    rows[row] = left + eventWidthPercent;

    positioned.push({ event, left, row });
  }

  return positioned;
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

// Find matching step_end for a step_start
function findStepPairs(events: LctlEvent[]): Map<number, number> {
  const pairs = new Map<number, number>();
  const stack: { seq: number; agent: string }[] = [];

  for (const event of events) {
    if (event.type === 'step_start') {
      stack.push({ seq: event.seq, agent: event.agent || '' });
    } else if (event.type === 'step_end' && stack.length > 0) {
      // Find matching start (same agent, LIFO order)
      for (let i = stack.length - 1; i >= 0; i--) {
        if (stack[i].agent === (event.agent || '')) {
          pairs.set(stack[i].seq, event.seq);
          stack.splice(i, 1);
          break;
        }
      }
    }
  }

  return pairs;
}

function ZoomControls() {
  const zoomLevel = useStore((s) => s.zoomLevel);
  const zoomIn = useStore((s) => s.zoomIn);
  const zoomOut = useStore((s) => s.zoomOut);
  const setZoomLevel = useStore((s) => s.setZoomLevel);

  return (
    <div className="zoom-controls">
      <button
        className="zoom-button"
        onClick={zoomOut}
        disabled={zoomLevel <= 0.25}
        title="Zoom out"
      >
        −
      </button>
      <span className="zoom-level" title="Click to reset">
        <button
          className="zoom-reset"
          onClick={() => setZoomLevel(1)}
        >
          {Math.round(zoomLevel * 100)}%
        </button>
      </span>
      <button
        className="zoom-button"
        onClick={zoomIn}
        disabled={zoomLevel >= 4}
        title="Zoom in"
      >
        +
      </button>
    </div>
  );
}

function SwimLaneView() {
  const chain = useStore((s) => s.chain);
  const visibleEvents = useVisibleEvents();
  const replay = useStore((s) => s.replay);
  const selectedEvent = useStore((s) => s.selectedEvent);
  const selectEvent = useStore((s) => s.selectEvent);
  const zoomLevel = useStore((s) => s.zoomLevel);

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

  // Find step pairs for linking
  const stepPairs = useMemo(() => {
    if (!chain?.events) return new Map();
    return findStepPairs(chain.events);
  }, [chain?.events]);

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
  const baseWidth = Math.max(800, maxSeq * 30);
  const width = baseWidth * zoomLevel;

  const getPosition = (seq: number) => {
    return ((seq - minSeq) / (maxSeq - minSeq || 1)) * 100;
  };

  // Layout events with collision detection per track
  const layoutCache = useMemo(() => {
    const cache = new Map<string, PositionedEvent[]>();
    for (const { agent, events } of tracks) {
      cache.set(agent, layoutEventsWithCollisionDetection(events, getPosition, width, zoomLevel));
    }
    return cache;
  }, [tracks, width, zoomLevel]);

  // Calculate max rows per track for height
  const getTrackHeight = (agent: string): number => {
    const positioned = layoutCache.get(agent) || [];
    const maxRow = positioned.reduce((max, p) => Math.max(max, p.row), 0);
    return (maxRow + 1) * 32 + 8; // 32px per row + padding
  };

  return (
    <div className="swim-lanes" style={{ minWidth: width }}>
      {/* Time axis */}
      <div className="swim-lane-axis" style={{ marginLeft: 100 }}>
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
        className="swim-lane-playhead"
        style={{
          left: `calc(100px + ${getPosition(replay.currentSeq)}% * ${zoomLevel})`,
        }}
      />

      {/* SVG layer for step linking connectors */}
      <svg
        className="step-connectors"
        style={{
          position: 'absolute',
          top: 24,
          left: 100,
          width: `${width}px`,
          height: '100%',
          pointerEvents: 'none',
          zIndex: 1
        }}
      >
        {tracks.map(({ agent }) => {
          const positioned = layoutCache.get(agent) || [];
          const trackOffset = tracks.slice(0, tracks.findIndex(t => t.agent === agent))
            .reduce((sum, t) => sum + getTrackHeight(t.agent), 24);

          return positioned.map(({ event, left, row }) => {
            if (event.type !== 'step_start') return null;
            const endSeq = stepPairs.get(event.seq);
            if (!endSeq) return null;

            const endPos = positioned.find(p => p.event.seq === endSeq);
            if (!endPos) return null;

            const startX = (left / 100) * width + EVENT_WIDTH / 2;
            const endX = (endPos.left / 100) * width + EVENT_WIDTH / 2;
            const y1 = trackOffset + row * 32 + 16;
            const y2 = trackOffset + endPos.row * 32 + 16;

            // Draw a curved connector
            const midX = (startX + endX) / 2;
            const curveY = Math.min(y1, y2) - 8;

            return (
              <path
                key={`link-${event.seq}-${endSeq}`}
                d={`M ${startX} ${y1} Q ${midX} ${curveY} ${endX} ${y2}`}
                fill="none"
                stroke="var(--color-step-start)"
                strokeWidth={1.5}
                strokeOpacity={0.4}
                strokeDasharray="4 2"
              />
            );
          });
        })}
      </svg>

      {/* Swim lanes */}
      {tracks.map(({ agent }) => {
        const positioned = layoutCache.get(agent) || [];
        const trackHeight = getTrackHeight(agent);

        return (
          <div key={agent} className="swim-lane" style={{ height: trackHeight }}>
            <div className="swim-lane-label">{agent === '_global' ? 'Global' : agent}</div>
            <div className="swim-lane-track" style={{ width, height: trackHeight - 8 }}>
              {positioned.map(({ event, left, row }) => {
                const isSelected = selectedEvent?.seq === event.seq;
                const hasLink = event.type === 'step_start' && stepPairs.has(event.seq);
                const isLinkedEnd = event.type === 'step_end' &&
                  Array.from(stepPairs.values()).includes(event.seq);

                return (
                  <div
                    key={event.seq}
                    className={`swim-lane-event ${event.type} ${hasLink || isLinkedEnd ? 'linked' : ''}`}
                    style={{
                      left: `${left}%`,
                      top: row * 32 + 4,
                      background: isSelected ? 'var(--vscode-accent)' : undefined,
                      borderColor: isSelected ? 'var(--vscode-accent)' : undefined
                    }}
                    onClick={() => selectEvent(event)}
                    title={`[${event.seq}] ${event.type}${event.agent ? ` (${event.agent})` : ''}`}
                  >
                    {EVENT_ICONS[event.type] || '•'}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
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
          {viewMode === 'swimlane' && <ZoomControls />}
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
