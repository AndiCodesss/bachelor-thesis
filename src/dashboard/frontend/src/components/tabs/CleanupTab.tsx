import type { RunStatus } from '../../types'

import { StopButton } from '../StopButton'

interface CleanupTabProps {
  force: boolean
  canStop: boolean
  status: RunStatus
  onForceChange: (value: boolean) => void
  onStart: () => void
  onStop: () => void
}

export function CleanupTab({ force, canStop, status, onForceChange, onStart, onStop }: CleanupTabProps) {
  return (
    <>
      <div>
        <h2 className="panel-title">System Cleanup</h2>
        <p className="panel-desc">Remove generated signals, reset states, and delete logs.</p>
      </div>

      <div className="form-group form-group--plain">
        <label>Danger Zone</label>
        <label className={`checkbox-label checkbox-label--spaced ${force ? 'active' : ''}`}>
          <input type="checkbox" checked={force} onChange={event => onForceChange(event.target.checked)} />
          Force Delete (Uncheck for Dry-Run)
        </label>
        <p className="help-text">
          If force delete is disabled, the script will only output exactly what directories and files <b>would</b> be deleted without actually mutating the filesystem.
        </p>
      </div>

      <div className="action-row">
        <button
          type="button"
          className={`action-btn action-btn--fill ${force ? 'danger' : ''}`}
          onClick={onStart}
          disabled={status === 'running'}
        >
          {status === 'running' ? <><span className="loader" />Running Cleanup...</> : (force ? 'Nuke Artifacts' : 'Dry Run Cleanup')}
        </button>
        {canStop && <StopButton onClick={onStop} />}
      </div>
    </>
  )
}
