import type { CacheConfig, RunStatus } from '../../types'

import { StopButton } from '../StopButton'

interface CacheTabProps {
  cacheConfig: CacheConfig | null
  splits: string[]
  session: string
  execMode: string
  barFilter: string
  clean: boolean
  status: RunStatus
  onSplitsChange: (updater: (prev: string[]) => string[]) => void
  onSessionChange: (value: string) => void
  onExecModeChange: (value: string) => void
  onBarFilterChange: (value: string) => void
  onCleanChange: (value: boolean) => void
  onStart: () => void
  onStop: () => void
}

export function CacheTab({
  cacheConfig,
  splits,
  session,
  execMode,
  barFilter,
  clean,
  status,
  onSplitsChange,
  onSessionChange,
  onExecModeChange,
  onBarFilterChange,
  onCleanChange,
  onStart,
  onStop,
}: CacheTabProps) {
  const toggleSplit = (split: string) => {
    onSplitsChange(prev => {
      if (split === 'all') {
        return ['all']
      }
      const next = prev.filter(item => item !== 'all')
      if (next.includes(split)) {
        return next.filter(item => item !== split)
      }
      return [...next, split]
    })
  }

  return (
    <>
      <div>
        <h2 className="panel-title">Cache Runner</h2>
        <p className="panel-desc">Rebuild feature matrices for ML.</p>
      </div>

      {cacheConfig ? (
        <>
          <div className="form-group">
            <label>Splits</label>
            <div className="checkbox-group">
              {cacheConfig.splits.map(split => (
                <label key={split} className={`checkbox-label ${splits.includes(split) ? 'active' : ''}`}>
                  <input type="checkbox" checked={splits.includes(split)} onChange={() => toggleSplit(split)} />
                  {split}
                </label>
              ))}
            </div>
          </div>
          <div className="form-group">
            <label>Session</label>
            <select value={session} onChange={event => onSessionChange(event.target.value)}>
              {cacheConfig.session_filters.map(sessionFilter => (
                <option key={sessionFilter} value={sessionFilter}>{sessionFilter}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Execution Mode</label>
            <select value={execMode} onChange={event => onExecModeChange(event.target.value)}>
              {cacheConfig.exec_modes.map(mode => (
                <option key={mode} value={mode}>{mode}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Bar Filter (Optional)</label>
            <select value={barFilter} onChange={event => onBarFilterChange(event.target.value)}>
              <option value="">All Bar Configs</option>
              {cacheConfig.bar_filters.map(filter => (
                <option key={filter} value={filter}>{filter}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Options</label>
            <label className={`checkbox-label ${clean ? 'active' : ''}`} style={{ width: 'fit-content' }}>
              <input type="checkbox" checked={clean} onChange={event => onCleanChange(event.target.checked)} />
              Clean Cache First
            </label>
          </div>
          <div style={{ display: 'flex', gap: '1rem', marginTop: 'auto' }}>
            <button className="action-btn" style={{ flex: 1 }} onClick={onStart} disabled={status === 'running' || splits.length === 0}>
              {status === 'running' ? <><span className="loader" />Running...</> : 'Execute Cache Builder'}
            </button>
            {status === 'running' && <StopButton onClick={onStop} />}
          </div>
        </>
      ) : (
        <p style={{ color: 'var(--text-muted)' }}>Loading config...</p>
      )}
    </>
  )
}
