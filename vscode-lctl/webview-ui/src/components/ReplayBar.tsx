import { useEffect, useRef } from 'react';
import { useStore } from '../store';

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4];

export default function ReplayBar() {
  const replay = useStore((s) => s.replay);
  const chain = useStore((s) => s.chain);
  const setReplayState = useStore((s) => s.setReplayState);
  const stepForward = useStore((s) => s.stepForward);
  const stepBackward = useStore((s) => s.stepBackward);
  const togglePlay = useStore((s) => s.togglePlay);
  const setPlaySpeed = useStore((s) => s.setPlaySpeed);

  const intervalRef = useRef<number | null>(null);

  // Auto-play logic
  useEffect(() => {
    if (replay.isPlaying && replay.currentSeq < replay.maxSeq) {
      intervalRef.current = window.setInterval(() => {
        const { currentSeq, maxSeq, isPlaying } = useStore.getState().replay;
        if (isPlaying && currentSeq < maxSeq) {
          stepForward();
        } else {
          setReplayState({ isPlaying: false });
        }
      }, 500 / replay.speed);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [replay.isPlaying, replay.speed, stepForward, setReplayState]);

  // Stop when reaching end
  useEffect(() => {
    if (replay.currentSeq >= replay.maxSeq && replay.isPlaying) {
      setReplayState({ isPlaying: false });
    }
  }, [replay.currentSeq, replay.maxSeq, replay.isPlaying, setReplayState]);

  if (!chain) return null;

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const seq = parseInt(e.target.value, 10);
    setReplayState({ currentSeq: seq });
  };

  const handleReset = () => {
    setReplayState({ currentSeq: 0, isPlaying: false });
  };

  const handleEnd = () => {
    setReplayState({ currentSeq: replay.maxSeq, isPlaying: false });
  };

  return (
    <div className="replay-bar">
      {/* Reset to start */}
      <button
        className="replay-button"
        onClick={handleReset}
        title="Go to start"
      >
        ⏮
      </button>

      {/* Step backward */}
      <button
        className="replay-button"
        onClick={stepBackward}
        disabled={replay.currentSeq <= 0}
        title="Step backward"
      >
        ◀
      </button>

      {/* Play/Pause */}
      <button
        className={`replay-button ${replay.isPlaying ? 'active' : ''}`}
        onClick={togglePlay}
        title={replay.isPlaying ? 'Pause' : 'Play'}
      >
        {replay.isPlaying ? '⏸' : '▶'}
      </button>

      {/* Step forward */}
      <button
        className="replay-button"
        onClick={stepForward}
        disabled={replay.currentSeq >= replay.maxSeq}
        title="Step forward"
      >
        ▶
      </button>

      {/* Go to end */}
      <button
        className="replay-button"
        onClick={handleEnd}
        title="Go to end"
      >
        ⏭
      </button>

      {/* Timeline slider */}
      <input
        type="range"
        className="replay-slider"
        min={0}
        max={replay.maxSeq}
        value={replay.currentSeq}
        onChange={handleSliderChange}
      />

      {/* Position display */}
      <span className="replay-position">
        {replay.currentSeq} / {replay.maxSeq}
      </span>

      {/* Speed selector */}
      <select
        value={replay.speed}
        onChange={(e) => setPlaySpeed(parseFloat(e.target.value))}
        style={{
          background: 'var(--vscode-input-background)',
          color: 'var(--vscode-foreground)',
          border: '1px solid var(--vscode-border)',
          borderRadius: 4,
          padding: '4px 8px',
          fontSize: 11
        }}
      >
        {SPEED_OPTIONS.map((speed) => (
          <option key={speed} value={speed}>
            {speed}x
          </option>
        ))}
      </select>
    </div>
  );
}
