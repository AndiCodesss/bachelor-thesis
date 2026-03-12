import { formatAutonomyTimestamp, formatUsd } from '../../lib/format'
import type { AutonomyStatus, RunStatus } from '../../types'

import { StopButton } from '../StopButton'

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

  return (
    <>
      <div>
        <h2 className="panel-title">Autonomy Overview</h2>
        <p className="panel-desc" style={{ marginBottom: '1rem' }}>Monitor experiments and control the discovery pipeline.</p>
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
            <div className="metric-card" style={{ gridColumn: '1 / -1' }}>
              <span className="metric-label">Financial Snapshot (Last {statusData.financial.tested} Experiments)</span>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem' }}>
                <div>
                  <span className="metric-sub">Avg Net PNL: </span>
                  <span className="metric-value" style={{ fontSize: '1.2rem' }}>${statusData.financial.avg_net_pnl.toFixed(2)}</span>
                </div>
                <div>
                  <span className="metric-sub">Pass Rate: </span>
                  <span className="metric-value" style={{ fontSize: '1.2rem' }}>{statusData.financial.pass_rate_pct.toFixed(1)}%</span>
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
            <div className="metric-card" style={{ gridColumn: '1 / -1' }}>
              <span className="metric-label">Recent Results</span>
              <div className="hyp-list" style={{ marginTop: '0.75rem' }}>
                {statusData.recent_results.map(result => {
                  const edgeRejected = Boolean(result.edge_status) && !['global_edge', 'disabled', 'selection_skip'].includes(result.edge_status)
                  const verdictClass = result.verdict === 'PASS' ? 'completed' : (result.verdict === 'FAIL' ? 'failed' : '')
                  const edgeNote = edgeRejected
                    ? `Edge discovery stopped the full backtest (${result.edge_status}).`
                    : 'Full backtest executed.'
                  const bestEdgeNote = result.best_horizon_bars !== null && typeof result.best_avg_trade_pnl === 'number'
                    ? ` Best edge horizon: ${result.best_horizon_bars} bars at ${formatUsd(result.best_avg_trade_pnl)} per trade.`
                    : ''
                  return (
                    <div
                      key={`${result.strategy}-${result.timestamp}`}
                      className="hyp-item"
                      style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.75rem', width: '100%' }}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem 0.75rem', minWidth: 0 }}>
                          <span className="metric-sub" style={{ color: 'var(--text-primary)' }}>{result.strategy}</span>
                          <span className="metric-sub">{result.bar}</span>
                          <span className="metric-sub">{formatAutonomyTimestamp(result.timestamp)}</span>
                        </div>
                        <span className={`status-indicator ${verdictClass}`} style={{ fontSize: '0.65rem', padding: '0.1rem 0.4rem' }}>
                          {result.verdict}
                        </span>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '0.5rem 1rem', width: '100%' }}>
                        <div>
                          <div className="metric-sub">Signals</div>
                          <div style={{ fontFamily: 'var(--font-mono)' }}>{result.signal_count ?? 'n/a'}</div>
                        </div>
                        <div>
                          <div className="metric-sub">Edge Events</div>
                          <div style={{ fontFamily: 'var(--font-mono)' }}>{result.edge_events ?? 'n/a'}</div>
                        </div>
                        <div>
                          <div className="metric-sub">Backtest Trades</div>
                          <div style={{ fontFamily: 'var(--font-mono)' }}>{result.backtest_trades ?? 'n/a'}</div>
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
            <div className="metric-card" style={{ gridColumn: '1 / -1' }}>
              <span className="metric-label">Active Hypotheses</span>
              <div className="hyp-list">
                {statusData.active_hypotheses.map(hypothesis => (
                  <div key={hypothesis.id} className="hyp-item">
                    <span className="metric-sub">{hypothesis.id}</span>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>{hypothesis.tasks} tasks</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="form-group" style={{ borderTop: '1px solid var(--border-dim)', paddingTop: '1.5rem' }}>
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
              onChange={event => {
                if (!event.target.checked) {
                  onValidatorOnlyChange(false)
                  onOrchestratorOnlyChange(true)
                } else {
                  onOrchestratorOnlyChange(false)
                }
              }}
            />
            Enable Validator
          </label>
          <label className={`checkbox-label ${!validatorOnly ? 'active' : ''}`}>
            <input
              type="checkbox"
              checked={!validatorOnly}
              onChange={event => {
                if (!event.target.checked) {
                  onOrchestratorOnlyChange(false)
                  onValidatorOnlyChange(true)
                } else {
                  onValidatorOnlyChange(false)
                }
              }}
            />
            Enable Orchestrator
          </label>
        </div>
      </div>

      <div className="form-group">
        <label>Orchestrator Lanes</label>
        <div className="checkbox-group">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(count => (
            <label
              key={count}
              className={`checkbox-label ${laneCount === count ? 'active' : ''}`}
              style={{
                cursor: orchestratorEnabled ? 'pointer' : 'not-allowed',
                minWidth: '2.2rem',
                justifyContent: 'center',
                opacity: orchestratorEnabled ? 1 : 0.45,
              }}
              onClick={() => {
                if (orchestratorEnabled) {
                  onLaneCountChange(count)
                }
              }}
            >
              {count}
            </label>
          ))}
        </div>
      </div>

      <div className="form-group">
        <label>Execution Flags</label>
        <div className="checkbox-group" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
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

      <div style={{ display: 'flex', gap: '1rem', marginTop: 'auto' }}>
        <button className="action-btn" style={{ flex: 1 }} onClick={onStart} disabled={status === 'running' || autonomySelectionInvalid}>
          {status === 'running' ? <><span className="loader" />Deploying Mission...</> : 'Launch Autonomy'}
        </button>
        {status === 'running' && <StopButton onClick={onStop} />}
      </div>
    </>
  )
}
