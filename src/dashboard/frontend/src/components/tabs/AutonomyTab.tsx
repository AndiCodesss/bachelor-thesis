import { formatAutonomyTimestamp, formatUsd } from '../../lib/format'
import type { AutonomyStatus, RunStatus } from '../../types'

import { StopButton } from '../StopButton'

const LANE_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

function verdictClass(verdict: string): string {
  if (verdict === 'PASS') {
    return 'completed'
  }
  if (verdict === 'FAIL') {
    return 'failed'
  }
  return ''
}

interface AutonomyTabProps {
  agentConfig: string
  allowBootstrap: boolean
  laneCount: number
  mission: string
  noResume: boolean
  orchestratorOnly: boolean
  status: RunStatus
  statusData: AutonomyStatus | null
  useNotebookLM: boolean
  validatorOnly: boolean
  onAgentConfigChange: (value: string) => void
  onAllowBootstrapChange: (value: boolean) => void
  onLaneCountChange: (value: number) => void
  onMissionChange: (value: string) => void
  onNoResumeChange: (value: boolean) => void
  onOrchestratorOnlyChange: (value: boolean) => void
  onStart: () => void
  onStop: () => void
  onUseNotebookLMChange: (value: boolean) => void
  onValidatorOnlyChange: (value: boolean) => void
}

export function AutonomyTab({
  agentConfig,
  allowBootstrap,
  laneCount,
  mission,
  noResume,
  orchestratorOnly,
  status,
  statusData,
  useNotebookLM,
  validatorOnly,
  onAgentConfigChange,
  onAllowBootstrapChange,
  onLaneCountChange,
  onMissionChange,
  onNoResumeChange,
  onOrchestratorOnlyChange,
  onStart,
  onStop,
  onUseNotebookLMChange,
  onValidatorOnlyChange,
}: AutonomyTabProps) {
  const orchestratorEnabled = !validatorOnly
  const autonomySelectionInvalid = validatorOnly && orchestratorOnly

  const setValidatorEnabled = (enabled: boolean) => {
    if (enabled) {
      onOrchestratorOnlyChange(false)
      return
    }
    onValidatorOnlyChange(false)
    onOrchestratorOnlyChange(true)
  }

  const setOrchestratorEnabled = (enabled: boolean) => {
    if (enabled) {
      onValidatorOnlyChange(false)
      return
    }
    onOrchestratorOnlyChange(false)
    onValidatorOnlyChange(true)
  }

  return (
    <>
      <div>
        <h2 className="panel-title">Autonomy Overview</h2>
        <p className="panel-desc panel-desc--compact">Monitor experiments and control the discovery pipeline.</p>
      </div>

      {statusData && (
        <div className="metrics-grid">
          <div className="metric-card">
            <span className="metric-label">Research Queue</span>
            <span className="metric-value">{statusData.queue.pending} <span className="metric-sub">pending</span></span>
            <span className="metric-sub">
              <span className="metric-highlight">{statusData.queue.completed} done</span> · <span className="metric-error">{statusData.queue.failed} failed</span>
            </span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Mission Budget</span>
            <span className="metric-value">{statusData.budget.experiments_run} / {statusData.budget.max_experiments}</span>
            <span className="metric-sub">
              failures: {Object.keys(statusData.budget.failures).length > 0
                ? Object.entries(statusData.budget.failures).map(([key, value]) => `${key}:${value}`).join(', ')
                : 'none'}
            </span>
          </div>
          {statusData.financial.tested > 0 && (
            <div className="metric-card metric-card--wide">
              <span className="metric-label">Financial Snapshot (Last {statusData.financial.tested} Experiments)</span>
              <div className="metrics-summary">
                <div className="metrics-summary-item">
                  <span className="metric-sub">Avg Net PNL: </span>
                  <span className="metric-value metric-value--medium">${statusData.financial.avg_net_pnl.toFixed(2)}</span>
                </div>
                <div className="metrics-summary-item">
                  <span className="metric-sub">Pass Rate: </span>
                  <span className="metric-value metric-value--medium">{statusData.financial.pass_rate_pct.toFixed(1)}%</span>
                </div>
              </div>
              <div className="hyp-list">
                {statusData.financial.best && (
                  <div className="hyp-item">
                    <span className="metric-sub">Best: {statusData.financial.best.strategy}</span>
                    <span className="metric-highlight">${statusData.financial.best.net_pnl.toFixed(2)} (SR: {statusData.financial.best.sharpe.toFixed(2)}, Trades: {statusData.financial.best.trades})</span>
                  </div>
                )}
                {statusData.financial.worst && (
                  <div className="hyp-item">
                    <span className="metric-sub">Worst: {statusData.financial.worst.strategy}</span>
                    <span className="metric-error">${statusData.financial.worst.net_pnl.toFixed(2)} (SR: {statusData.financial.worst.sharpe.toFixed(2)}, Trades: {statusData.financial.worst.trades})</span>
                  </div>
                )}
              </div>
            </div>
          )}
          {statusData.recent_results.length > 0 && (
            <div className="metric-card metric-card--wide">
              <span className="metric-label">Recent Results</span>
              <div className="hyp-list hyp-list--spacious">
                {statusData.recent_results.map(result => {
                  const edgeRejected = Boolean(result.edge_status) && !['global_edge', 'disabled', 'selection_skip'].includes(result.edge_status)
                  const edgeNote = edgeRejected
                    ? `Edge discovery stopped the full backtest (${result.edge_status}).`
                    : 'Full backtest executed.'
                  const bestEdgeNote = result.best_horizon_bars !== null && typeof result.best_avg_trade_pnl === 'number'
                    ? ` Best edge horizon: ${result.best_horizon_bars} bars at ${formatUsd(result.best_avg_trade_pnl)} per trade.`
                    : ''
                  return (
                    <div
                      key={`${result.strategy}-${result.timestamp}`}
                      className="hyp-item hyp-item--stacked"
                    >
                      <div className="hyp-item-header">
                        <div className="hyp-item-meta">
                          <span className="metric-sub metric-sub--primary">{result.strategy}</span>
                          <span className="metric-sub">{result.bar}</span>
                          <span className="metric-sub">{formatAutonomyTimestamp(result.timestamp)}</span>
                        </div>
                        <span className={`status-indicator status-indicator--compact ${verdictClass(result.verdict)}`}>
                          {result.verdict}
                        </span>
                      </div>
                      <div className="result-stats-grid">
                        <div>
                          <div className="metric-sub">Signals</div>
                          <div className="mono-text">{result.signal_count ?? 'n/a'}</div>
                        </div>
                        <div>
                          <div className="metric-sub">Edge Events</div>
                          <div className="mono-text">{result.edge_events ?? 'n/a'}</div>
                        </div>
                        <div>
                          <div className="metric-sub">Backtest Trades</div>
                          <div className="mono-text">{result.backtest_trades ?? 'n/a'}</div>
                        </div>
                        <div>
                          <div className="metric-sub">Net PNL</div>
                          <div className={typeof result.net_pnl === 'number' && result.net_pnl >= 0 ? 'metric-highlight' : 'metric-error'}>
                            {formatUsd(result.net_pnl)}
                          </div>
                        </div>
                      </div>
                      <span className="metric-sub">
                        {edgeNote}
                        {bestEdgeNote}
                        {result.failure_code && result.failure_code !== result.edge_status ? ` Failure: ${result.failure_code}.` : ''}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
          {statusData.active_hypotheses.length > 0 && (
            <div className="metric-card metric-card--wide">
              <span className="metric-label">Active Hypotheses</span>
              <div className="hyp-list">
                {statusData.active_hypotheses.map(hypothesis => (
                  <div key={hypothesis.id} className="hyp-item">
                    <span className="metric-sub">{hypothesis.id}</span>
                    <span className="mono-text">{hypothesis.tasks} tasks</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="form-group form-group--top-divider">
        <label>Mission Config Path</label>
        <input type="text" value={mission} onChange={event => onMissionChange(event.target.value)} />
      </div>
      <div className="form-group">
        <label>Agent Config Path</label>
        <input type="text" value={agentConfig} onChange={event => onAgentConfigChange(event.target.value)} />
      </div>

      <div className="form-group">
        <label>Worker Toggles</label>
        <div className="checkbox-group">
          <label className={`checkbox-label ${!orchestratorOnly ? 'active' : ''}`}>
            <input
              type="checkbox"
              checked={!orchestratorOnly}
              onChange={event => setValidatorEnabled(event.target.checked)}
            />
            Enable Validator
          </label>
          <label className={`checkbox-label ${!validatorOnly ? 'active' : ''}`}>
            <input
              type="checkbox"
              checked={!validatorOnly}
              onChange={event => setOrchestratorEnabled(event.target.checked)}
            />
            Enable Orchestrator
          </label>
        </div>
      </div>

      <div className="form-group">
        <label>Orchestrator Lanes</label>
        <div className="lane-selector" role="group" aria-label="Orchestrator lanes">
          {LANE_OPTIONS.map(count => (
            <button
              key={count}
              type="button"
              className={`checkbox-label lane-button ${laneCount === count ? 'active' : ''}`}
              onClick={() => onLaneCountChange(count)}
              disabled={!orchestratorEnabled}
              aria-pressed={laneCount === count}
            >
              {count}
            </button>
          ))}
        </div>
      </div>

      <div className="form-group">
        <label>Execution Flags</label>
        <div className="checkbox-group toggle-group--stacked">
          <label className={`checkbox-label ${noResume ? 'active' : ''}`}>
            <input type="checkbox" checked={noResume} onChange={event => onNoResumeChange(event.target.checked)} />
            Fresh State (Reset Runtime State)
          </label>
          <label className={`checkbox-label ${allowBootstrap ? 'active' : ''}`}>
            <input type="checkbox" checked={allowBootstrap} onChange={event => onAllowBootstrapChange(event.target.checked)} />
            Allow Bootstrap
          </label>
          <label className={`checkbox-label ${useNotebookLM ? 'active' : ''}`}>
            <input type="checkbox" checked={useNotebookLM} onChange={event => onUseNotebookLMChange(event.target.checked)} />
            Use NotebookLM Research
          </label>
        </div>
      </div>

      <div className="action-row">
        <button
          type="button"
          className="action-btn action-btn--fill"
          onClick={onStart}
          disabled={status === 'running' || autonomySelectionInvalid}
        >
          {status === 'running' ? <><span className="loader" />Deploying Mission...</> : 'Launch Autonomy'}
        </button>
        {status === 'running' && <StopButton onClick={onStop} />}
      </div>
    </>
  )
}
