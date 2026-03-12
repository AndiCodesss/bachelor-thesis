import { useEffect, useRef } from 'react'

import type { RunStatus } from '../types'

interface TerminalPanelProps {
  currentCmd: string
  logs: string[]
  status: RunStatus
}

export function TerminalPanel({
  currentCmd,
  logs,
  status,
}: TerminalPanelProps) {
  const terminalRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="panel terminal-panel">
      <div className="terminal-header">
        <div className="terminal-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="4 17 10 11 4 5" />
            <line x1="12" y1="19" x2="20" y2="19" />
          </svg>
          {currentCmd || 'Output Terminal'}
        </div>
        <div className="terminal-controls">
          <div className={`status-indicator ${status}`}>
            <div className="status-dot" />
            {status}
          </div>
        </div>
      </div>

      <div className="terminal-body" ref={terminalRef}>
        {logs.length > 0 ? (
          logs.map((log, idx) => (
            <div key={idx} className="terminal-line">{log}</div>
          ))
        ) : (
          <div className="terminal-empty">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeOpacity="0.5">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            <p>System idle. Ready for tasks.</p>
          </div>
        )}
      </div>
    </div>
  )
}
