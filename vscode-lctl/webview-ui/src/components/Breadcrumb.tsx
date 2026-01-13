import { useStore } from '../store';

export default function Breadcrumb() {
  const navigationHistory = useStore((s) => s.navigationHistory);
  const selectedEvent = useStore((s) => s.selectedEvent);
  const navigateBack = useStore((s) => s.navigateBack);
  const clearNavigation = useStore((s) => s.clearNavigation);
  const selectEvent = useStore((s) => s.selectEvent);

  if (!selectedEvent && navigationHistory.length === 0) {
    return null;
  }

  return (
    <div className="breadcrumb-bar">
      {navigationHistory.length > 0 && (
        <button
          className="breadcrumb-back"
          onClick={navigateBack}
          title="Go back (Alt+Left)"
        >
          ←
        </button>
      )}

      <div className="breadcrumb-trail">
        {navigationHistory.map((item, index) => (
          <span key={`${item.event.seq}-${index}`} className="breadcrumb-item">
            <button
              className="breadcrumb-link"
              onClick={() => {
                // Navigate to this item, removing subsequent history
                const newHistory = navigationHistory.slice(0, index);
                useStore.setState({ navigationHistory: newHistory });
                selectEvent(item.event);
              }}
            >
              [{item.event.seq}] {item.label}
            </button>
            <span className="breadcrumb-separator">›</span>
          </span>
        ))}

        {selectedEvent && (
          <span className="breadcrumb-current">
            [{selectedEvent.seq}] {selectedEvent.agent ? `${selectedEvent.agent}: ` : ''}{selectedEvent.type}
          </span>
        )}
      </div>

      {navigationHistory.length > 0 && (
        <button
          className="breadcrumb-clear"
          onClick={clearNavigation}
          title="Clear navigation history"
        >
          ✕
        </button>
      )}
    </div>
  );
}
