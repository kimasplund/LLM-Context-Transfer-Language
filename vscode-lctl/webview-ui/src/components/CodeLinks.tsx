import { vscode } from '../App';
import type { LctlEvent } from '../types';

interface FileLink {
  path: string;
  line?: number;
  label: string;
}

// Extract file paths from various event data patterns
function extractFileLinks(event: LctlEvent): FileLink[] {
  const links: FileLink[] = [];
  const data = event.data;

  if (!data) return links;

  // Common path patterns in tool calls
  const pathFields = [
    'file_path', 'path', 'source_file', 'target_file',
    'old_file', 'new_file', 'filename'
  ];

  for (const field of pathFields) {
    const value = data[field];
    if (typeof value === 'string' && value.length > 0) {
      const line = data.line || data.line_number || data.start_line;
      links.push({
        path: value,
        line: typeof line === 'number' ? line : undefined,
        label: field === 'file_path' ? value : `${field}: ${value}`
      });
    }
  }

  // Extract from tool-specific patterns
  if (event.type === 'tool_call' && data.tool) {
    const tool = String(data.tool);

    // Read tool
    if (tool === 'Read' || tool === 'read') {
      const input = data.input as Record<string, unknown> | undefined;
      if (input?.file_path) {
        links.push({
          path: String(input.file_path),
          line: input.offset ? Number(input.offset) : undefined,
          label: `Read: ${input.file_path}`
        });
      }
    }

    // Edit/MultiEdit tools
    if (tool === 'Edit' || tool === 'MultiEdit') {
      const input = data.input as Record<string, unknown> | undefined;
      if (input?.file_path) {
        links.push({
          path: String(input.file_path),
          label: `Edit: ${input.file_path}`
        });
      }
    }

    // Write tool
    if (tool === 'Write') {
      const input = data.input as Record<string, unknown> | undefined;
      if (input?.file_path) {
        links.push({
          path: String(input.file_path),
          label: `Write: ${input.file_path}`
        });
      }
    }

    // Glob tool - show matched files
    if (tool === 'Glob') {
      const output = data.output;
      if (Array.isArray(output)) {
        for (const file of output.slice(0, 5)) {
          if (typeof file === 'string') {
            links.push({ path: file, label: file });
          }
        }
      }
    }

    // Grep tool - show files with matches
    if (tool === 'Grep') {
      const output = data.output;
      if (typeof output === 'object' && output !== null) {
        const files = Object.keys(output as Record<string, unknown>).slice(0, 5);
        for (const file of files) {
          links.push({ path: file, label: file });
        }
      }
    }
  }

  // Deduplicate by path
  const seen = new Set<string>();
  return links.filter(link => {
    if (seen.has(link.path)) return false;
    seen.add(link.path);
    return true;
  });
}

// Extract git-related information
function extractGitInfo(event: LctlEvent): { commit?: string; branch?: string } | null {
  const data = event.data;
  if (!data) return null;

  // Check for git commit in tool call
  if (event.type === 'tool_call') {
    const tool = String(data.tool || '');
    if (tool === 'Bash') {
      const input = data.input as Record<string, unknown> | undefined;
      const command = String(input?.command || '');
      // Check if it's a git commit command
      if (command.includes('git commit')) {
        const output = String(data.output || '');
        // Try to extract commit hash from output like "[main abc1234] Commit message"
        const match = output.match(/\[[\w-]+\s+([a-f0-9]{7,40})\]/);
        if (match) {
          return { commit: match[1] };
        }
      }
    }
  }

  // Check data fields directly
  if (data.commit_hash || data.commit) {
    return { commit: String(data.commit_hash || data.commit) };
  }

  return null;
}

interface CodeLinksProps {
  event: LctlEvent;
}

export default function CodeLinks({ event }: CodeLinksProps) {
  const fileLinks = extractFileLinks(event);
  const gitInfo = extractGitInfo(event);

  if (fileLinks.length === 0 && !gitInfo) {
    return null;
  }

  const handleOpenFile = (path: string, line?: number) => {
    vscode.postMessage({
      type: 'openFile',
      path,
      line
    });
  };

  return (
    <div style={{ marginTop: 12 }}>
      {fileLinks.length > 0 && (
        <div>
          <div className="details-field-label" style={{ marginBottom: 8 }}>
            File References
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {fileLinks.map((link, i) => (
              <button
                key={i}
                onClick={() => handleOpenFile(link.path, link.line)}
                style={{
                  background: 'var(--vscode-input-background)',
                  border: '1px solid var(--vscode-border)',
                  borderRadius: 4,
                  padding: '4px 8px',
                  color: 'var(--vscode-textLink-foreground)',
                  cursor: 'pointer',
                  textAlign: 'left',
                  fontSize: 11,
                  fontFamily: 'var(--vscode-editor-font-family, monospace)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6
                }}
              >
                <span>📄</span>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {link.label}
                  {link.line && <span style={{ opacity: 0.6 }}>:{link.line}</span>}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {gitInfo && (
        <div style={{ marginTop: fileLinks.length > 0 ? 12 : 0 }}>
          <div className="details-field-label" style={{ marginBottom: 8 }}>
            Git Info
          </div>
          <div style={{
            background: 'var(--vscode-input-background)',
            borderRadius: 4,
            padding: 8,
            fontSize: 11,
            fontFamily: 'monospace'
          }}>
            {gitInfo.commit && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>🔗</span>
                <span>Commit: {gitInfo.commit}</span>
              </div>
            )}
            {gitInfo.branch && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                <span>🌿</span>
                <span>Branch: {gitInfo.branch}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
