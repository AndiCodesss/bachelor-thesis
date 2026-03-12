import type { RunStatus } from '../../types'

import { StopButton } from '../StopButton'

interface CleanupTabProps {
  force: boolean
  status: RunStatus
  onForceChange: (value: boolean) => void
  onStart: () => void
  onStop: () => void
}

export function CleanupTab({ force, status, onForceChange, onStart, onStop }: CleanupTabProps) {
  return (
    <>
      <div>
        <h2 className="panel-title">System Cleanup</h2>
        <p className="panel-desc">Remove generated signals, reset states, and delete logs.</p>
      </div>

      <div className="form-group" style={{ border: 'none' }}>
        <label>Danger Zone</label>
        <label className={`checkbox-label ${force ? 'active' : ''}`} style={{ marginTop: '0.5rem' }}>
          <input type="checkbox" checked={force} onChange={event => onForceChange(event.target.checked)} />
          Force Delete (Uncheck for Dry-Run)
        </label>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: '1.4' }}>
          If force delete is disabled, the script will only output exactly what directories and files <b>would</b> be deleted without actually mutating the filesystem.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginTop: 'auto' }}>
        <button className={`action-btn ${force ? 'danger' : ''}`} style={{ flex: 1 }} onClick={onStart} disabled={status === 'running'}>
          {status === 'running' ? <><span className="loader" />Running Cleanup...</> : (force ? 'Nuke Artifacts' : 'Dry Run Cleanup')}
        </button>
        {status === 'running' && <StopButton onClick={onStop} />}
      </div>
    </>
  )
}
