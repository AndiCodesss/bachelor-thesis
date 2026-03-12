import { formatMetricKey, formatMetricValue } from '../../lib/format'
import type { GauntletItem, SignalDetails, SignalItem } from '../../types'

interface SignalsTabProps {
  selectedSignal: string | null
  signalDetails: SignalDetails | null
  signalsList: SignalItem[]
  onSelectSignal: (strategy: string) => void
}

export function SignalsTab({
  selectedSignal,
  signalDetails,
  signalsList,
  onSelectSignal,
}: SignalsTabProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div>
        <h2 className="panel-title">Signals Explorer</h2>
        <p className="panel-desc" style={{ marginBottom: '1rem' }}>Inspect the generated files and detailed verification constraints of every tested strategy.</p>
      </div>

      <div className="explorer-container">
        <div className="explorer-sidebar">
          {signalsList.length > 0 ? signalsList.map(signal => (
            <button
              key={signal.strategy}
              className={`signal-list-item ${selectedSignal === signal.strategy ? 'selected' : ''}`}
              onClick={() => onSelectSignal(signal.strategy)}
            >
              <div className="signal-header">
                <span className="signal-name">{signal.strategy}</span>
                <span className={`status-indicator ${signal.verdict === 'PASS' ? 'completed' : (signal.verdict === 'FAIL' ? 'failed' : '')}`} style={{ fontSize: '0.65rem', padding: '0.1rem 0.4rem' }}>
                  {signal.verdict}
                </span>
              </div>
              <span className="metric-sub">{signal.timestamp} | {signal.bar_config}</span>
            </button>
          )) : (
            <p className="metric-sub">No recent signals found in logs.</p>
          )}
        </div>

        <div className="explorer-main">
          {signalDetails ? (
            <>
              <div className="signal-code-view">
                <div style={{ color: 'var(--accent-primary)', marginBottom: '1rem', borderBottom: '1px solid var(--border-dim)', paddingBottom: '0.5rem' }}>
                  // File: research/signals/{signalDetails.strategy}.py
                </div>
                {signalDetails.code}
              </div>
              <div className="signal-metrics-view">
                <div className="metric-card">
                  <span className="metric-label">All Financial Metrics</span>
                  <div className="hyp-list" style={{ marginTop: '0.75rem' }}>
                    {Object.keys(signalDetails.metrics).length > 0 ? (
                      Object.entries(signalDetails.metrics).map(([key, value]) => (
                        <div key={key} className="hyp-item" style={{ display: 'flex', justifyContent: 'space-between', width: '100%', padding: '0.3rem 0' }}>
                          <span className="metric-sub">{formatMetricKey(key)}</span>
                          <span className="metric-value" style={{ fontSize: '0.9rem' }}>{formatMetricValue(key, value)}</span>
                        </div>
                      ))
                    ) : (
                      <span className="metric-sub">No metrics available</span>
                    )}
                  </div>
                </div>

                <div className="metric-card">
                  <span className="metric-label">Gauntlet Verification</span>
                  <span
                    className="metric-value"
                    style={{
                      fontSize: '1.2rem',
                      color: signalDetails.gauntlet?.overall_verdict === 'PASS' ? 'var(--success)' : 'var(--danger)',
                    }}
                  >
                    {typeof signalDetails.gauntlet?.overall_verdict === 'string'
                      ? signalDetails.gauntlet.overall_verdict
                      : 'No Data'}
                  </span>

                  <div className="hyp-list" style={{ marginTop: '1rem' }}>
                    {Object.entries(signalDetails.gauntlet).map(([key, value]) => {
                      if (key === 'overall_verdict' || key === 'pass_count' || key === 'total_tests') {
                        return null
                      }
                      if (typeof value !== 'object' || value === null) {
                        return null
                      }
                      const item = value as GauntletItem
                      const passed = item.verdict === 'PASS'
                      return (
                        <div key={key} className="hyp-item" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                            <span className="metric-sub">{key}</span>
                            <span className={passed ? 'gauntlet-pass' : 'gauntlet-fail'}>{item.verdict}</span>
                          </div>
                          <span style={{ fontSize: '0.7rem', color: 'var(--border-focus)', marginTop: '0.2rem' }}>{item.msg}</span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="terminal-empty" style={{ flex: 1, border: '1px dashed var(--border-dim)', borderRadius: 'var(--radius-md)' }}>
              <p>Select a signal from the sidebar to inspect</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
