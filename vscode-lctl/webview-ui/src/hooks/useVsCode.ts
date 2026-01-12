import type { VsCodeApi, WebviewMessage } from '../types';

// Cached VS Code API instance
let vsCodeApi: VsCodeApi | null = null;

export function getVsCodeApi(): VsCodeApi {
  if (!vsCodeApi) {
    // Try to get the VS Code API (only available in webview context)
    if (typeof acquireVsCodeApi === 'function') {
      // Cast to our type since the actual API is compatible
      vsCodeApi = acquireVsCodeApi() as unknown as VsCodeApi;
    } else {
      // In development/browser mode, provide a mock
      vsCodeApi = {
        postMessage: (message: WebviewMessage) => {
          console.log('[VSCode Mock] postMessage:', message);
        },
        getState: <T>() => {
          const state = localStorage.getItem('lctl-webview-state');
          return state ? JSON.parse(state) as T : undefined;
        },
        setState: <T>(state: T) => {
          localStorage.setItem('lctl-webview-state', JSON.stringify(state));
        }
      };
    }
  }
  return vsCodeApi!;
}
