import { useEffect, useRef } from 'react'

import type { RunStatus, ThinkerData } from '../types'

interface TerminalPanelProps {
  activeTab: 'cache' | 'autonomy' | 'cleanup'
  currentCmd: string
  logs: string[]
  status: RunStatus
  thinkerData: ThinkerData | null
  thinkerView: 'logs' | 'thinker'
  onThinkerViewChange: (view: 'logs' | 'thinker') => void
}

export function TerminalPanel({
  activeTab,
  currentCmd,
  logs,
  status,
  thinkerData,
  thinkerView,
  onThinkerViewChange,
}: TerminalPanelProps) {
  const terminalRef = useRef<HTMLDivElement>(null)
  const thinkerFeedRef = useRef<HTMLDivElement>(null)
  const thinkerMode = activeTab === 'autonomy' && thinkerView === 'thinker'

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs])

  useEffect(() => {
    if (thinkerFeedRef.current) {
      thinkerFeedRef.current.scrollTop = thinkerFeedRef.current.scrollHeight
    }
  }, [thinkerData])

  const title = !thinkerMode
    ? (currentCmd || 'Output Terminal')
    : `Thinker · ${thinkerData?.session_id?.slice(0, 8) ?? '...'}`

  return (
    <div className="panel terminal-panel">
      <div className="terminal-header">
        <div className="terminal-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="4 17 10 11 4 5" />
            <line x1="12" y1="19" x2="20" y2="19" />
          </svg>
          {title}
        </div>
        <div className="terminal-controls">
          {activeTab === 'autonomy' && (
            <div className="thinker-view-toggle">
              <button
                type="button"
                className={`thinker-toggle-btn ${thinkerView === 'logs' ? 'active' : ''}`}
                onClick={() => onThinkerViewChange('logs')}
                aria-pressed={thinkerView === 'logs'}
              >
                Logs
              </button>
              <button
                type="button"
                className={`thinker-toggle-btn ${thinkerView === 'thinker' ? 'active' : ''}`}
                onClick={() => onThinkerViewChange('thinker')}
                aria-pressed={thinkerView === 'thinker'}
              >
                Thinker
                {thinkerData?.is_active && <span className="thinker-active-dot" />}
              </button>
            </div>
          )}
          <div className={`status-indicator ${status}`}>
            <div className="status-dot" />
            {status}
          </div>
        </div>
      </div>

      {thinkerMode ? (
        <div className="thinker-feed" ref={thinkerFeedRef}>
          {thinkerData && thinkerData.events.length > 0 ? (
            thinkerData.events.map((event, idx) => {
              if (event.type === 'text') {
                return (
                  <div key={idx} className="thinker-event thinker-event-text">
                    {event.content}
                  </div>
                )
              }
              if (event.type === 'tool_call') {
                return (
                  <div key={idx} className="thinker-event thinker-event-tool">
                    <span className="thinker-tool-badge">{event.tool}</span>
                    <span className="thinker-summary">{event.summary}</span>
                  </div>
                )
              }
              if (event.type === 'tool_result') {
                return (
                  <div key={idx} className="thinker-event thinker-event-result">
                    {event.tool && (
                      <span className="thinker-tool-badge thinker-tool-badge--result">
                        {event.tool}
                      </span>
                    )}
                    {event.summary}
                  </div>
                )
              }
              return null
            })
          ) : (
            <div className="terminal-empty">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeOpacity="0.5">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v4l3 3" />
              </svg>
              <p>{thinkerData ? 'No thinker activity yet.' : 'Loading thinker data...'}</p>
            </div>
          )}
        </div>
      ) : (
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
      )}
    </div>
  )
}
