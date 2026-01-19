import { useEffect, useState, useCallback } from 'react';
import { useStore } from '../store';

interface Shortcut {
  key: string;
  description: string;
  modifiers?: ('ctrl' | 'alt' | 'shift' | 'meta')[];
}

const SHORTCUTS: Shortcut[] = [
  { key: 'Space', description: 'Play/Pause replay' },
  { key: '←', description: 'Previous event' },
  { key: '→', description: 'Next event' },
  { key: '↑', description: 'First event' },
  { key: '↓', description: 'Last event' },
  { key: 'Escape', description: 'Deselect event' },
  { key: '?', description: 'Show shortcuts' },
  { key: '+', description: 'Zoom in' },
  { key: '-', description: 'Zoom out' },
  { key: '0', description: 'Reset zoom' },
  { key: '←', modifiers: ['alt'], description: 'Navigate back' },
  { key: 'f', modifiers: ['ctrl'], description: 'Focus search' },
];

function formatShortcut(shortcut: Shortcut): string {
  const parts: string[] = [];
  if (shortcut.modifiers?.includes('ctrl')) parts.push('Ctrl');
  if (shortcut.modifiers?.includes('alt')) parts.push('Alt');
  if (shortcut.modifiers?.includes('shift')) parts.push('Shift');
  if (shortcut.modifiers?.includes('meta')) parts.push('Cmd');
  parts.push(shortcut.key);
  return parts.join(' + ');
}

interface KeyboardShortcutsProps {
  isOpen: boolean;
  onClose: () => void;
}

export function KeyboardShortcutsOverlay({ isOpen, onClose }: KeyboardShortcutsProps) {
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="keyboard-shortcuts-overlay" onClick={onClose}>
      <div className="keyboard-shortcuts-panel" onClick={(e) => e.stopPropagation()}>
        <div className="shortcuts-header">
          <span className="shortcuts-title">Keyboard Shortcuts</span>
          <button className="shortcuts-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="shortcuts-list">
          {SHORTCUTS.map((shortcut, index) => (
            <div key={index} className="shortcut-item">
              <span className="shortcut-desc">{shortcut.description}</span>
              <span className="shortcut-key">{formatShortcut(shortcut)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function useKeyboardShortcuts() {
  const [showShortcuts, setShowShortcuts] = useState(false);
  const chain = useStore((s) => s.chain);
  const replay = useStore((s) => s.replay);
  const setReplayState = useStore((s) => s.setReplayState);
  const togglePlay = useStore((s) => s.togglePlay);
  const selectEvent = useStore((s) => s.selectEvent);
  const selectedEvent = useStore((s) => s.selectedEvent);
  const navigateBack = useStore((s) => s.navigateBack);
  const zoomIn = useStore((s) => s.zoomIn);
  const zoomOut = useStore((s) => s.zoomOut);
  const setZoomLevel = useStore((s) => s.setZoomLevel);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Don't trigger shortcuts when typing in input fields
    if (
      e.target instanceof HTMLInputElement ||
      e.target instanceof HTMLTextAreaElement
    ) {
      return;
    }

    const events = chain?.events || [];
    const maxSeq = events.length;

    switch (e.key) {
      case ' ':
        e.preventDefault();
        togglePlay();
        break;

      case 'ArrowLeft':
        if (e.altKey) {
          e.preventDefault();
          navigateBack();
        } else {
          e.preventDefault();
          const prevSeq = Math.max(0, replay.currentSeq - 1);
          setReplayState({ currentSeq: prevSeq });
          const prevEvent = events[prevSeq - 1];
          if (prevEvent) selectEvent(prevEvent);
        }
        break;

      case 'ArrowRight':
        e.preventDefault();
        const nextSeq = Math.min(maxSeq, replay.currentSeq + 1);
        setReplayState({ currentSeq: nextSeq });
        const nextEvent = events[nextSeq - 1];
        if (nextEvent) selectEvent(nextEvent);
        break;

      case 'ArrowUp':
        e.preventDefault();
        setReplayState({ currentSeq: 0 });
        selectEvent(null);
        break;

      case 'ArrowDown':
        e.preventDefault();
        setReplayState({ currentSeq: maxSeq });
        const lastEvent = events[maxSeq - 1];
        if (lastEvent) selectEvent(lastEvent);
        break;

      case 'Escape':
        e.preventDefault();
        selectEvent(null);
        setShowShortcuts(false);
        break;

      case '?':
        e.preventDefault();
        setShowShortcuts((prev) => !prev);
        break;

      case '+':
      case '=':
        e.preventDefault();
        zoomIn();
        break;

      case '-':
        e.preventDefault();
        zoomOut();
        break;

      case '0':
        e.preventDefault();
        setZoomLevel(1);
        break;

      case 'f':
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          // Focus search input if it exists
          const searchInput = document.querySelector('.search-input') as HTMLInputElement;
          if (searchInput) {
            searchInput.focus();
          }
        }
        break;

      case 'j':
        // Jump to next event of same type
        if (selectedEvent) {
          e.preventDefault();
          const currentIndex = events.findIndex((ev) => ev.seq === selectedEvent.seq);
          for (let i = currentIndex + 1; i < events.length; i++) {
            if (events[i].type === selectedEvent.type) {
              selectEvent(events[i]);
              setReplayState({ currentSeq: events[i].seq });
              break;
            }
          }
        }
        break;

      case 'k':
        // Jump to previous event of same type
        if (selectedEvent) {
          e.preventDefault();
          const currentIndex = events.findIndex((ev) => ev.seq === selectedEvent.seq);
          for (let i = currentIndex - 1; i >= 0; i--) {
            if (events[i].type === selectedEvent.type) {
              selectEvent(events[i]);
              setReplayState({ currentSeq: events[i].seq });
              break;
            }
          }
        }
        break;

      case 'e':
        // Jump to next error
        e.preventDefault();
        const errorIndex = events.findIndex(
          (ev, i) => ev.type === 'error' && i > replay.currentSeq - 1
        );
        if (errorIndex !== -1) {
          selectEvent(events[errorIndex]);
          setReplayState({ currentSeq: events[errorIndex].seq });
        }
        break;
    }
  }, [
    chain,
    replay.currentSeq,
    selectedEvent,
    togglePlay,
    setReplayState,
    selectEvent,
    navigateBack,
    zoomIn,
    zoomOut,
    setZoomLevel,
  ]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return {
    showShortcuts,
    setShowShortcuts,
  };
}
