import { create } from 'zustand';
import type { LctlChainFile, LctlEvent, ChainStats, ReplayState, FilterState, FactInfo } from './types';

interface NavigationItem {
  event: LctlEvent;
  label: string;
}

interface AppState {
  // Chain data
  chain: LctlChainFile | null;
  chainPath: string | null;
  isRecording: boolean;

  // Comparison chain for diff view
  compareChain: LctlChainFile | null;
  comparePath: string | null;
  isDiffMode: boolean;

  // Computed stats
  stats: ChainStats | null;

  // Replay state
  replay: ReplayState;

  // Filter state
  filter: FilterState;

  // Selected event
  selectedEvent: LctlEvent | null;

  // Navigation history for breadcrumbs
  navigationHistory: NavigationItem[];

  // Zoom level for timeline (1.0 = 100%)
  zoomLevel: number;

  // Details panel height (pixels)
  detailsPanelHeight: number;

  // Actions
  setChain: (chain: LctlChainFile, path: string, isRecording: boolean) => void;
  setReplayState: (state: Partial<ReplayState>) => void;
  setFilter: (filter: Partial<FilterState>) => void;
  selectEvent: (event: LctlEvent | null) => void;
  stepForward: () => void;
  stepBackward: () => void;
  togglePlay: () => void;
  setPlaySpeed: (speed: number) => void;
  setCompareChain: (chain: LctlChainFile | null, path: string | null) => void;
  toggleDiffMode: () => void;

  // New navigation actions
  navigateBack: () => void;
  clearNavigation: () => void;
  setZoomLevel: (level: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  setDetailsPanelHeight: (height: number) => void;
}

// Model pricing per 1M tokens
const MODEL_PRICING: Record<string, { input: number; output: number }> = {
  'claude-opus-4.5': { input: 5.0, output: 25.0 },
  'claude-sonnet-4': { input: 3.0, output: 15.0 },
  'claude-haiku-3': { input: 0.25, output: 1.25 },
  'gpt-4': { input: 30.0, output: 60.0 },
  'gpt-4-turbo': { input: 10.0, output: 30.0 },
  'gpt-3.5-turbo': { input: 0.5, output: 1.5 },
  'default': { input: 3.0, output: 15.0 },
};

function computeStats(chain: LctlChainFile): ChainStats {
  const events = chain.events || [];
  const agents = new Set<string>();
  const tools = new Set<string>();
  const factsMap = new Map<string, FactInfo>();

  let errorCount = 0;
  let warningCount = 0;
  let totalDurationMs = 0;
  let tokensIn = 0;
  let tokensOut = 0;

  for (const event of events) {
    if (event.agent) agents.add(event.agent);

    if (event.type === 'error') errorCount++;

    if (event.type === 'tool_call' && event.data?.tool) {
      tools.add(event.data.tool as string);
    }

    if (event.type === 'step_end' && event.data) {
      totalDurationMs += (event.data.duration_ms as number) || 0;
      tokensIn += (event.data.tokens_in as number) || 0;
      tokensOut += (event.data.tokens_out as number) || 0;
    }

    if (event.type === 'fact_added' || event.type === 'fact_modified') {
      const id = event.data?.id as string;
      if (id) {
        factsMap.set(id, {
          id,
          text: (event.data?.text as string) || '',
          confidence: (event.data?.confidence as number) || 0,
          source: event.agent,
          lastModified: event.timestamp
        });
      }
    }
  }

  const pricing = MODEL_PRICING['default'];
  const estimatedCost =
    (tokensIn / 1_000_000) * pricing.input +
    (tokensOut / 1_000_000) * pricing.output;

  return {
    eventCount: events.length,
    errorCount,
    warningCount,
    totalDurationMs,
    tokensIn,
    tokensOut,
    estimatedCost,
    agents: Array.from(agents),
    tools: Array.from(tools),
    facts: Array.from(factsMap.values())
  };
}

export const useStore = create<AppState>((set, get) => ({
  // Initial state
  chain: null,
  chainPath: null,
  isRecording: false,
  compareChain: null,
  comparePath: null,
  isDiffMode: false,
  stats: null,
  replay: {
    isPlaying: false,
    currentSeq: 0,
    speed: 1,
    maxSeq: 0
  },
  filter: {
    eventTypes: [],
    agents: [],
    searchQuery: '',
    timeRange: null
  },
  selectedEvent: null,
  navigationHistory: [],
  zoomLevel: 1.0,
  detailsPanelHeight: 250,

  // Actions
  setChain: (chain, path, isRecording) => {
    const stats = computeStats(chain);
    const maxSeq = chain.events?.length || 0;
    set({
      chain,
      chainPath: path,
      isRecording,
      stats,
      replay: {
        ...get().replay,
        maxSeq,
        currentSeq: maxSeq
      }
    });
  },

  setReplayState: (state) => {
    set({ replay: { ...get().replay, ...state } });
  },

  setFilter: (filter) => {
    set({ filter: { ...get().filter, ...filter } });
  },

  selectEvent: (event) => {
    const { selectedEvent, navigationHistory } = get();

    // Add current selection to history before changing (if there was one)
    if (selectedEvent && event && selectedEvent.seq !== event.seq) {
      const label = selectedEvent.agent
        ? `${selectedEvent.agent}: ${selectedEvent.type}`
        : selectedEvent.type;
      const newHistory = [...navigationHistory, { event: selectedEvent, label }];
      // Keep last 10 items
      set({
        selectedEvent: event,
        navigationHistory: newHistory.slice(-10),
        replay: { ...get().replay, currentSeq: event.seq }
      });
    } else {
      set({ selectedEvent: event });
      if (event) {
        set({ replay: { ...get().replay, currentSeq: event.seq } });
      }
    }
  },

  stepForward: () => {
    const { replay } = get();
    if (replay.currentSeq < replay.maxSeq) {
      set({ replay: { ...replay, currentSeq: replay.currentSeq + 1 } });
    }
  },

  stepBackward: () => {
    const { replay } = get();
    if (replay.currentSeq > 0) {
      set({ replay: { ...replay, currentSeq: replay.currentSeq - 1 } });
    }
  },

  togglePlay: () => {
    const { replay } = get();
    set({ replay: { ...replay, isPlaying: !replay.isPlaying } });
  },

  setPlaySpeed: (speed) => {
    set({ replay: { ...get().replay, speed } });
  },

  setCompareChain: (chain, path) => {
    set({ compareChain: chain, comparePath: path });
  },

  toggleDiffMode: () => {
    set({ isDiffMode: !get().isDiffMode });
  },

  // Navigation actions
  navigateBack: () => {
    const { navigationHistory } = get();
    if (navigationHistory.length === 0) return;

    const newHistory = [...navigationHistory];
    const lastItem = newHistory.pop();
    if (lastItem) {
      set({
        selectedEvent: lastItem.event,
        navigationHistory: newHistory,
        replay: { ...get().replay, currentSeq: lastItem.event.seq }
      });
    }
  },

  clearNavigation: () => {
    set({ navigationHistory: [], selectedEvent: null });
  },

  // Zoom actions
  setZoomLevel: (level) => {
    set({ zoomLevel: Math.max(0.25, Math.min(4, level)) });
  },

  zoomIn: () => {
    const { zoomLevel } = get();
    set({ zoomLevel: Math.min(4, zoomLevel * 1.25) });
  },

  zoomOut: () => {
    const { zoomLevel } = get();
    set({ zoomLevel: Math.max(0.25, zoomLevel / 1.25) });
  },

  // Panel resize
  setDetailsPanelHeight: (height) => {
    set({ detailsPanelHeight: Math.max(100, Math.min(500, height)) });
  }
}));

// Selector for filtered events
export function useFilteredEvents(): LctlEvent[] {
  const chain = useStore((s) => s.chain);
  const filter = useStore((s) => s.filter);

  if (!chain?.events) return [];

  return chain.events.filter((event) => {
    // Filter by event type
    if (filter.eventTypes.length > 0 && !filter.eventTypes.includes(event.type)) {
      return false;
    }

    // Filter by agent
    if (filter.agents.length > 0 && event.agent && !filter.agents.includes(event.agent)) {
      return false;
    }

    // Filter by search query
    if (filter.searchQuery) {
      const query = filter.searchQuery.toLowerCase();
      const matchesType = event.type.toLowerCase().includes(query);
      const matchesAgent = event.agent?.toLowerCase().includes(query);
      const matchesData = JSON.stringify(event.data || {}).toLowerCase().includes(query);
      if (!matchesType && !matchesAgent && !matchesData) {
        return false;
      }
    }

    return true;
  });
}

// Selector for visible events based on replay position
export function useVisibleEvents(): LctlEvent[] {
  const events = useFilteredEvents();
  const currentSeq = useStore((s) => s.replay.currentSeq);
  return events.filter((e) => e.seq <= currentSeq);
}

// Selector for timeline tracks (grouped by agent)
export function useTimelineTracks(): Array<{ agent: string; events: LctlEvent[] }> {
  const events = useVisibleEvents();
  const trackMap = new Map<string, LctlEvent[]>();

  for (const event of events) {
    const agent = event.agent || '_global';
    if (!trackMap.has(agent)) {
      trackMap.set(agent, []);
    }
    trackMap.get(agent)!.push(event);
  }

  return Array.from(trackMap.entries())
    .map(([agent, events]) => ({ agent, events }))
    .sort((a, b) => {
      // Global track first, then alphabetically
      if (a.agent === '_global') return -1;
      if (b.agent === '_global') return 1;
      return a.agent.localeCompare(b.agent);
    });
}
