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
  const gauntletVerdict = typeof signalDetails?.gauntlet?.overall_verdict === 'string'
    ? signalDetails.gauntlet.overall_verdict
    : null

  return (
    <div className="signal-explorer">
      <div>
        <h2 className="panel-title">Signals Explorer</h2>
        <p className="panel-desc panel-desc--compact">Inspect the generated files and detailed verification constraints of every tested strategy.</p>
      </div>

      <div className="explorer-container">
        <div className="explorer-sidebar">
          {signalsList.length > 0 ? signalsList.map(signal => (
            <button
              key={signal.strategy}
              type="button"
              className={`signal-list-item ${selectedSignal === signal.strategy ? 'selected' : ''}`}
              onClick={() => onSelectSignal(signal.strategy)}
            >
              <div className="signal-header">
                <span className="signal-name">{signal.strategy}</span>
                <span className={`status-indicator status-indicator--compact ${signal.verdict === 'PASS' ? 'completed' : (signal.verdict === 'FAIL' ? 'failed' : '')}`}>
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
                <div className="signal-file-header">
                  // File: research/signals/{signalDetails.strategy}.py
                </div>
                {signalDetails.code}
              </div>
              <div className="signal-metrics-view">
                <div className="metric-card">
                  <span className="metric-label">All Financial Metrics</span>
                  <div className="hyp-list hyp-list--spacious">
                    {Object.keys(signalDetails.metrics).length > 0 ? (
                      Object.entries(signalDetails.metrics).map(([key, value]) => (
                        <div key={key} className="hyp-item metric-list-row">
                          <span className="metric-sub">{formatMetricKey(key)}</span>
                          <span className="metric-value metric-value--small">{formatMetricValue(key, value)}</span>
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
                    className={`metric-value gauntlet-verdict ${gauntletVerdict === 'PASS' ? 'gauntlet-pass' : 'gauntlet-fail'}`}
                  >
                    {gauntletVerdict ?? 'No Data'}
                  </span>

                  <div className="hyp-list hyp-list--block">
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
                        <div key={key} className="hyp-item hyp-item--stacked">
                          <div className="metric-list-row">
                            <span className="metric-sub">{key}</span>
                            <span className={passed ? 'gauntlet-pass' : 'gauntlet-fail'}>{item.verdict}</span>
                          </div>
                          <span className="gauntlet-msg">{item.msg}</span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="terminal-empty empty-state--framed">
              <p>Select a signal from the sidebar to inspect</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
