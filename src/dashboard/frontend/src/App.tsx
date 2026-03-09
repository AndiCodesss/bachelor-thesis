import { useEffect, useState, useRef } from 'react'

const API_URL = 'http://localhost:8000/api'
const WS_URL = 'ws://localhost:8000/ws'

type TabType = 'cache' | 'autonomy' | 'cleanup' | 'signals'
type RunStatus = 'idle' | 'running' | 'completed' | 'failed'

interface CacheConfig {
  splits: string[]
  session_filters: string[]
  exec_modes: string[]
  bar_filters: string[]
}

interface AutonomyStatus {
  queue: { pending: number; in_progress: number; completed: number; failed: number }
  budget: { experiments_run: number; max_experiments: number | "n/a"; failures: Record<string, number> }
  financial: {
    tested: number; avg_net_pnl: number; avg_sharpe: number; pass_rate_pct: number;
    best: { strategy: string; net_pnl: number; sharpe: number; trades: number } | null;
    worst: { strategy: string; net_pnl: number; sharpe: number; trades: number } | null;
  }
  active_hypotheses: Array<{ id: string; tasks: number }>
}

interface SignalItem {
  strategy: string;
  verdict: string;
  timestamp: string;
  bar_config: string;
}

interface SignalDetails {
  strategy: string;
  code: string;
  metrics: Record<string, unknown>;
  gauntlet: Record<string, unknown>;
  verdict: string;
  timestamp: string;
}

interface GauntletItem {
  verdict?: string;
  msg?: string;
}

interface ThinkerEvent {
  type: 'text' | 'tool_call' | 'tool_result'
  content?: string
  tool?: string
  summary?: string
}

interface ThinkerData {
  events: ThinkerEvent[]
  session_id: string | null
  is_active: boolean
  last_updated: string | null
}

function formatMetricKey(key: string): string {
  return key
    .split('_')
    .map(word => (word ? word.charAt(0).toUpperCase() + word.slice(1) : ''))
    .join(' ')
}

function formatMetricValue(key: string, value: unknown): string {
  if (value === null || value === undefined) {
    return 'N/A'
  }

  if (typeof value === 'number') {
    if (key.includes('pnl')) {
      return `$${Number(value).toFixed(2)}`
    }
    if (key === 'win_rate') {
      return `${(Number(value) * 100).toFixed(2)}%`
    }
    if (key.endsWith('_pct')) {
      return `${Number(value).toFixed(2)}%`
    }
    if (Number.isInteger(value) && !key.includes('ratio')) {
      return String(value)
    }
    return Math.abs(value) < 0.01 && value !== 0
      ? value.toExponential(4)
      : Number(value).toFixed(4)
  }

  if (typeof value === 'boolean') {
    return value ? 'true' : 'false'
  }

  if (typeof value === 'object') {
    return JSON.stringify(value)
  }

  return String(value)
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('cache')
  
  // Shared Run State
  const [runId, setRunId] = useState<string | null>(null)
  const [status, setStatus] = useState<RunStatus>('idle')
  const [logs, setLogs] = useState<string[]>([])
  const [currentCmd, setCurrentCmd] = useState<string>('')
  
  const terminalRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Cache State
  const [cacheConfig, setCacheConfig] = useState<CacheConfig | null>(null)
  const [cSplits, setCSplits] = useState<string[]>(['all'])
  const [cSession, setCSession] = useState('eth')
  const [cExecMode, setCExecMode] = useState('auto')
  const [cBarFilter, setCBarFilter] = useState('')
  const [cClean, setCClean] = useState(false)

  // Autonomy State
  const [aMission, setAMission] = useState('configs/missions/alpha-discovery.yaml')
  const [aAgent, setAAgent] = useState('configs/agents/llm_orchestrator.yaml')
  const [aNoResume, setANoResume] = useState(false)
  const [aAllowBootstrap, setAAllowBootstrap] = useState(false)
  const [aUseNotebookLM, setAUseNotebookLM] = useState(true)
  const [aValidatorOnly, setAValidatorOnly] = useState(false)
  const [aOrchestratorOnly, setAOrchestratorOnly] = useState(false)
  const [aLaneCount, setALaneCount] = useState(2)
  const [aStatusData, setAStatusData] = useState<AutonomyStatus | null>(null)
  const [thinkerView, setThinkerView] = useState<'logs' | 'thinker'>('logs')
  const [thinkerData, setThinkerData] = useState<ThinkerData | null>(null)
  const thinkerFeedRef = useRef<HTMLDivElement>(null)

  // Cleanup State
  const [clForce, setClForce] = useState(false)

  // Signals State
  const [signalsList, setSignalsList] = useState<SignalItem[]>([])
  const [selectedSignal, setSelectedSignal] = useState<string | null>(null)
  const [signalDetails, setSignalDetails] = useState<SignalDetails | null>(null)
  const orchestratorEnabled = !aValidatorOnly
  const autonomySelectionInvalid = aValidatorOnly && aOrchestratorOnly

  useEffect(() => {
    fetch(`${API_URL}/config/cache`)
      .then(r => r.json())
      .then(data => {
        setCacheConfig(data)
        if (data.session_filters.length > 0) setCSession(data.session_filters[0])
        if (data.exec_modes.length > 0) setCExecMode(data.exec_modes[0])
      })
      .catch(e => console.error("Failed to fetch cache config:", e))
      
    // Poll Autonomy status
    const pollAutonomy = () => {
      if (activeTab === 'autonomy') {
        fetch(`${API_URL}/autonomy/status`)
          .then(r => r.json())
          .then(data => setAStatusData(data))
          .catch(e => console.error("Failed to fetch autonomy status", e))

        fetch(`${API_URL}/autonomy/thinker`)
          .then(r => r.json())
          .then(data => setThinkerData(data))
          .catch(e => console.error("Failed to fetch thinker data", e))
      }
    }
    pollAutonomy()
    const interval = setInterval(pollAutonomy, 3000)
    
    // Fetch signals if tab clicked
    if (activeTab === 'signals') {
      fetch(`${API_URL}/signals`)
        .then(r => r.json())
        .then(data => setSignalsList(data))
        .catch(e => console.error("Failed to fetch signals array", e))
    }
    
    return () => clearInterval(interval)
  }, [activeTab])

  useEffect(() => {
    if (selectedSignal) {
      setSignalDetails(null)
      fetch(`${API_URL}/signals/${selectedSignal}`)
        .then(r => r.json())
        .then(data => setSignalDetails(data))
        .catch(e => console.error("Failed to fetch signal details", e))
    }
  }, [selectedSignal])

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

  const connectWebSocket = (id: string, cmdStr: string) => {
    setRunId(id)
    setCurrentCmd(cmdStr)
    if (wsRef.current) wsRef.current.close()
    
    const ws = new WebSocket(`${WS_URL}/runs/${id}/logs`)
    wsRef.current = ws
    
    ws.onmessage = (event) => setLogs(prev => [...prev, event.data])
    ws.onclose = () => {
      fetch(`${API_URL}/runs/${id}`)
        .then(r => r.json())
        .then(d => setStatus(d.status))
    }
  }

  const triggerRun = async (endpoint: string, payload: Record<string, unknown>) => {
    if (status === 'running') return
    setLogs([])
    setStatus('running')
    
    try {
      const resp = await fetch(`${API_URL}/run/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      const data = await resp.json()
      if (!resp.ok) {
        throw new Error(String(data.detail ?? data.error ?? `Request failed with status ${resp.status}`))
      }
      if (!data.run_id || !data.cmd) {
        throw new Error('Backend response missing run metadata.')
      }
      connectWebSocket(data.run_id, data.cmd)
    } catch (e) {
      console.error(e)
      setStatus('failed')
      setLogs([e instanceof Error ? e.message : 'Failed to connect to backend server.'])
    }
  }

  const stopRun = async () => {
    if (!runId || status !== 'running') return
    try {
      await fetch(`${API_URL}/runs/${runId}/stop`, { method: 'POST' })
      setStatus('failed')
    } catch(e) {
      console.error("Failed to stop run:", e)
    }
  }

  const startCache = () => triggerRun('cache', {
    splits: cSplits, session_filter: cSession, exec_mode: cExecMode, bar_filter: cBarFilter || null, clean: cClean
  })

  const startAutonomy = () => triggerRun('autonomy', {
    mission: aMission, agent_config: aAgent, no_resume: aNoResume,
    allow_bootstrap: aAllowBootstrap, use_notebooklm: aUseNotebookLM,
    validator_only: aValidatorOnly, orchestrator_only: aOrchestratorOnly,
    lane_count: aLaneCount
  })

  const startCleanup = () => triggerRun('cleanup', { force: clForce })

  const toggleSplit = (split: string) => {
    setCSplits(prev => {
      if (split === 'all') return ['all']
      const newSplits = prev.filter(s => s !== 'all')
      if (newSplits.includes(split)) return newSplits.filter(s => s !== split)
      return [...newSplits, split]  
    })
  }

  return (
    <div className="container">
      <header>
        <div className="header-left">
          <h1>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
            NQ-Alpha Core
          </h1>
          <div className="tabs">
            <button className={`tab-btn ${activeTab === 'cache' ? 'active' : ''}`} onClick={() => setActiveTab('cache')}>
              Cache Builder
            </button>
            <button className={`tab-btn ${activeTab === 'autonomy' ? 'active' : ''}`} onClick={() => setActiveTab('autonomy')}>
              Autonomy Control
            </button>
            <button className={`tab-btn ${activeTab === 'cleanup' ? 'active' : ''}`} onClick={() => setActiveTab('cleanup')}>
              System Cleanup
            </button>
            <button className={`tab-btn ${activeTab === 'signals' ? 'active' : ''}`} onClick={() => setActiveTab('signals')}>
              Signals Explorer
            </button>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="panel config-panel" style={activeTab === 'signals' ? {flex: 1, maxWidth: 'none', overflow: 'hidden'} : {}}>
          
          {/* TAB: CACHE BUILDER */}
          {activeTab === 'cache' && (
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
                      {cacheConfig.splits.map(s => (
                        <label key={s} className={`checkbox-label ${cSplits.includes(s) ? 'active' : ''}`}>
                          <input type="checkbox" checked={cSplits.includes(s)} onChange={() => toggleSplit(s)} />
                          {s}
                        </label>
                      ))}
                    </div>
                  </div>
                  <div className="form-group">
                    <label>Session</label>
                    <select value={cSession} onChange={e => setCSession(e.target.value)}>
                      {cacheConfig.session_filters.map(sf => <option key={sf} value={sf}>{sf}</option>)}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Execution Mode</label>
                    <select value={cExecMode} onChange={e => setCExecMode(e.target.value)}>
                      {cacheConfig.exec_modes.map(em => <option key={em} value={em}>{em}</option>)}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Bar Filter (Optional)</label>
                    <select value={cBarFilter} onChange={e => setCBarFilter(e.target.value)}>
                      <option value="">All Bar Configs</option>
                      {cacheConfig.bar_filters.map(bf => <option key={bf} value={bf}>{bf}</option>)}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Options</label>
                    <label className={`checkbox-label ${cClean ? 'active' : ''}`} style={{width: 'fit-content'}}>
                      <input type="checkbox" checked={cClean} onChange={e => setCClean(e.target.checked)}/>
                      Clean Cache First
                    </label>
                  </div>
                  <div style={{display: 'flex', gap: '1rem', marginTop: 'auto'}}>
                    <button className="action-btn" style={{flex: 1}} onClick={startCache} disabled={status === 'running' || cSplits.length === 0}>
                      {status === 'running' ? <><span className="loader"></span>Running...</> : 'Execute Cache Builder'}
                    </button>
                    {status === 'running' && activeTab === 'cache' && (
                      <button className="action-btn danger" onClick={stopRun} style={{padding: '0.8rem 1.5rem'}}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>
                      </button>
                    )}
                  </div>
                </>
              ) : <p style={{color: 'var(--text-muted)'}}>Loading config...</p>}
            </>
          )}

          {/* TAB: AUTONOMY CONTROL */}
          {activeTab === 'autonomy' && (
            <>
              <div>
                <h2 className="panel-title">Autonomy Overview</h2>
                <p className="panel-desc" style={{marginBottom: '1rem'}}>Monitor experiments and control the discovery pipeline.</p>
              </div>

              {aStatusData && (
                <div className="metrics-grid">
                  <div className="metric-card">
                    <span className="metric-label">Research Queue</span>
                    <span className="metric-value">{aStatusData.queue.pending} <span className="metric-sub">pending</span></span>
                    <span className="metric-sub">
                      <span className="metric-highlight">{aStatusData.queue.completed} done</span> · <span className="metric-error">{aStatusData.queue.failed} failed</span>
                    </span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Mission Budget</span>
                    <span className="metric-value">{aStatusData.budget.experiments_run} / {aStatusData.budget.max_experiments}</span>
                    <span className="metric-sub">failures: {Object.keys(aStatusData.budget.failures).length > 0 ? Object.entries(aStatusData.budget.failures).map(([k,v]) => `${k}:${v}`).join(', ') : 'none'}</span>
                  </div>
                  {aStatusData.financial.tested > 0 && (
                    <div className="metric-card" style={{gridColumn: '1 / -1'}}>
                      <span className="metric-label">Financial Snapshot (Last {aStatusData.financial.tested} Experiments)</span>
                      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem'}}>
                        <div>
                          <span className="metric-sub">Avg Net PNL: </span>
                          <span className="metric-value" style={{fontSize: '1.2rem'}}>${aStatusData.financial.avg_net_pnl.toFixed(2)}</span>
                        </div>
                        <div>
                          <span className="metric-sub">Pass Rate: </span>
                          <span className="metric-value" style={{fontSize: '1.2rem'}}>{aStatusData.financial.pass_rate_pct.toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="hyp-list">
                        {aStatusData.financial.best && (
                          <div className="hyp-item">
                            <span className="metric-sub">Best: {aStatusData.financial.best.strategy}</span>
                            <span className="metric-highlight">${aStatusData.financial.best.net_pnl.toFixed(2)} (SR: {aStatusData.financial.best.sharpe.toFixed(2)})</span>
                          </div>
                        )}
                        {aStatusData.financial.worst && (
                          <div className="hyp-item">
                            <span className="metric-sub">Worst: {aStatusData.financial.worst.strategy}</span>
                            <span className="metric-error">${aStatusData.financial.worst.net_pnl.toFixed(2)} (SR: {aStatusData.financial.worst.sharpe.toFixed(2)})</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  {aStatusData.active_hypotheses.length > 0 && (
                    <div className="metric-card" style={{gridColumn: '1 / -1'}}>
                      <span className="metric-label">Active Hypotheses</span>
                      <div className="hyp-list">
                        {aStatusData.active_hypotheses.map(h => (
                          <div key={h.id} className="hyp-item">
                            <span className="metric-sub">{h.id}</span>
                            <span style={{fontFamily: 'var(--font-mono)'}}>{h.tasks} tasks</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="form-group" style={{borderTop: '1px solid var(--border-dim)', paddingTop: '1.5rem'}}>
                <label>Mission Config Path</label>
                <input type="text" value={aMission} onChange={e => setAMission(e.target.value)} />
              </div>
              <div className="form-group">
                <label>Agent Config Path</label>
                <input type="text" value={aAgent} onChange={e => setAAgent(e.target.value)} />
              </div>
              
              <div className="form-group">
                <label>Worker Toggles</label>
                <div className="checkbox-group">
                  <label className={`checkbox-label ${!aOrchestratorOnly ? 'active' : ''}`}>
                    <input type="checkbox" checked={!aOrchestratorOnly} onChange={e => {
                      if (!e.target.checked) { setAValidatorOnly(false); setAOrchestratorOnly(true); }
                      else { setAOrchestratorOnly(false) }
                    }}/> Enable Validator
                  </label>
                  <label className={`checkbox-label ${!aValidatorOnly ? 'active' : ''}`}>
                    <input type="checkbox" checked={!aValidatorOnly} onChange={e => {
                      if (!e.target.checked) { setAOrchestratorOnly(false); setAValidatorOnly(true); }
                      else { setAValidatorOnly(false) }
                    }}/> Enable Orchestrator
                  </label>
                </div>
              </div>

              <div className="form-group">
                <label>Orchestrator Lanes</label>
                <div className="checkbox-group">
                  {[1,2,3,4,5,6,7,8,9,10].map(n => (
                    <label
                      key={n}
                      className={`checkbox-label ${aLaneCount === n ? 'active' : ''}`}
                      style={{
                        cursor: orchestratorEnabled ? 'pointer' : 'not-allowed',
                        minWidth: '2.2rem',
                        justifyContent: 'center',
                        opacity: orchestratorEnabled ? 1 : 0.45,
                      }}
                      onClick={() => {
                        if (orchestratorEnabled) {
                          setALaneCount(n)
                        }
                      }}
                    >
                      {n}
                    </label>
                  ))}
                </div>
              </div>

              <div className="form-group">
                <label>Execution Flags</label>
                <div className="checkbox-group" style={{flexDirection: 'column', alignItems: 'flex-start'}}>
                  <label className={`checkbox-label ${aNoResume ? 'active' : ''}`}>
                    <input type="checkbox" checked={aNoResume} onChange={e => setANoResume(e.target.checked)}/> Fresh State (Reset Runtime State)
                  </label>
                  <label className={`checkbox-label ${aAllowBootstrap ? 'active' : ''}`}>
                    <input type="checkbox" checked={aAllowBootstrap} onChange={e => setAAllowBootstrap(e.target.checked)}/> Allow Bootstrap
                  </label>
                  <label className={`checkbox-label ${aUseNotebookLM ? 'active' : ''}`}>
                    <input type="checkbox" checked={aUseNotebookLM} onChange={e => setAUseNotebookLM(e.target.checked)}/> Use NotebookLM Research
                  </label>
                </div>
              </div>

              <div style={{display: 'flex', gap: '1rem', marginTop: 'auto'}}>
                <button className="action-btn" style={{flex: 1}} onClick={startAutonomy} disabled={status === 'running' || autonomySelectionInvalid}>
                  {status === 'running' ? <><span className="loader"></span>Deploying Mission...</> : 'Launch Autonomy'}
                </button>
                {status === 'running' && activeTab === 'autonomy' && (
                  <button className="action-btn danger" onClick={stopRun} style={{padding: '0.8rem 1.5rem'}}>
                     <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>
                  </button>
                )}
              </div>
            </>
          )}

          {/* TAB: CLEANUP */}
          {activeTab === 'cleanup' && (
            <>
              <div>
                <h2 className="panel-title">System Cleanup</h2>
                <p className="panel-desc">Remove generated signals, reset states, and delete logs.</p>
              </div>

              <div className="form-group" style={{border: 'none'}}>
                <label>Danger Zone</label>
                <label className={`checkbox-label ${clForce ? 'active' : ''}`} style={{marginTop: '0.5rem'}}>
                  <input type="checkbox" checked={clForce} onChange={e => setClForce(e.target.checked)}/> 
                  Force Delete (Uncheck for Dry-Run)
                </label>
                <p style={{fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: '1.4'}}>
                  If force delete is disabled, the script will only output exactly what directories and files <b>would</b> be deleted without actually mutating the filesystem.
                </p>
              </div>

              <div style={{display: 'flex', gap: '1rem', marginTop: 'auto'}}>
                <button className={`action-btn ${clForce ? 'danger' : ''}`} style={{flex: 1}} onClick={startCleanup} disabled={status === 'running'}>
                  {status === 'running' ? <><span className="loader"></span>Running Cleanup...</> : (clForce ? 'Nuke Artifacts' : 'Dry Run Cleanup')}
                </button>
                {status === 'running' && activeTab === 'cleanup' && (
                  <button className="action-btn danger" onClick={stopRun} style={{padding: '0.8rem 1.5rem'}}>
                     <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>
                  </button>
                )}
              </div>
            </>
          )}

          {/* TAB: SIGNALS EXPLORER */}
          {activeTab === 'signals' && (
            <div style={{display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0}}>
              <div>
                <h2 className="panel-title">Signals Explorer</h2>
                <p className="panel-desc" style={{marginBottom: '1rem'}}>Inspect the generated files and detailed verification constraints of every tested strategy.</p>
              </div>

              <div className="explorer-container">
                <div className="explorer-sidebar">
                  {signalsList.length > 0 ? signalsList.map(s => (
                    <button 
                      key={s.strategy} 
                      className={`signal-list-item ${selectedSignal === s.strategy ? 'selected' : ''}`}
                      onClick={() => setSelectedSignal(s.strategy)}
                    >
                      <div className="signal-header">
                        <span className="signal-name">{s.strategy}</span>
                        <span className={`status-indicator ${s.verdict === 'PASS' ? 'completed' : (s.verdict === 'FAIL' ? 'failed' : '')}`} style={{fontSize: '0.65rem', padding: '0.1rem 0.4rem'}}>
                           {s.verdict}
                        </span>
                      </div>
                      <span className="metric-sub">{s.timestamp} | {s.bar_config}</span>
                    </button>
                  )) : <p className="metric-sub">No recent signals found in logs.</p>}
                </div>

                <div className="explorer-main">
                  {signalDetails ? (
                    <>
                      <div className="signal-code-view">
                        <div style={{color: 'var(--accent-primary)', marginBottom: '1rem', borderBottom: '1px solid var(--border-dim)', paddingBottom: '0.5rem'}}>
                          // File: research/signals/{signalDetails.strategy}.py
                        </div>
                        {signalDetails.code}
                      </div>
                      <div className="signal-metrics-view">
                        
                        <div className="metric-card">
                          <span className="metric-label">All Financial Metrics</span>
                          <div className="hyp-list" style={{marginTop: '0.75rem'}}>
                            {signalDetails.metrics && Object.keys(signalDetails.metrics).length > 0 ? (
                              Object.entries(signalDetails.metrics).map(([k, v]) => (
                                <div key={k} className="hyp-item" style={{display: 'flex', justifyContent: 'space-between', width: '100%', padding: '0.3rem 0'}}>
                                  <span className="metric-sub">{formatMetricKey(k)}</span>
                                  <span className="metric-value" style={{fontSize: '0.9rem'}}>{formatMetricValue(k, v)}</span>
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
                              color: signalDetails.gauntlet?.overall_verdict === 'PASS' ? 'var(--success)' : 'var(--danger)'
                            }}
                          >
                            {typeof signalDetails.gauntlet?.overall_verdict === 'string'
                              ? signalDetails.gauntlet.overall_verdict
                              : 'No Data'}
                          </span>
                          
                          <div className="hyp-list" style={{marginTop: '1rem'}}>
                             {signalDetails.gauntlet && Object.entries(signalDetails.gauntlet).map(([k, v]) => {
                               if (k === 'overall_verdict' || k === 'pass_count' || k === 'total_tests') return null;
                               if (typeof v !== 'object' || v === null) return null;
                               const item = v as GauntletItem;
                               const vpass = item.verdict === 'PASS';
                               return (
                                 <div key={k} className="hyp-item" style={{flexDirection: 'column', alignItems: 'flex-start'}}>
                                    <div style={{display: 'flex', justifyContent: 'space-between', width: '100%'}}>
                                      <span className="metric-sub">{k}</span>
                                      <span className={vpass ? 'gauntlet-pass' : 'gauntlet-fail'}>{item.verdict}</span>
                                    </div>
                                    <span style={{fontSize: '0.7rem', color: 'var(--border-focus)', marginTop: '0.2rem'}}>{item.msg}</span>
                                 </div>
                               )
                             })}
                          </div>
                        </div>

                      </div>
                    </>
                  ) : (
                    <div className="terminal-empty" style={{flex: 1, border: '1px dashed var(--border-dim)', borderRadius: 'var(--radius-md)'}}>
                       <p>Select a signal from the sidebar to inspect</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

        </div>

        {/* TERMINAL VIEW */}
        {activeTab !== 'signals' && (
        <div className="panel terminal-panel">
          <div className="terminal-header">
            <div className="terminal-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>
              {thinkerView === 'logs' || activeTab !== 'autonomy' ? (currentCmd || "Output Terminal") : `Thinker · ${thinkerData?.session_id?.slice(0, 8) ?? '...'}`}
            </div>
            <div style={{display: 'flex', alignItems: 'center', gap: '0.75rem'}}>
              {activeTab === 'autonomy' && (
                <div className="thinker-view-toggle">
                  <button
                    className={`thinker-toggle-btn ${thinkerView === 'logs' ? 'active' : ''}`}
                    onClick={() => setThinkerView('logs')}
                  >
                    Logs
                  </button>
                  <button
                    className={`thinker-toggle-btn ${thinkerView === 'thinker' ? 'active' : ''}`}
                    onClick={() => setThinkerView('thinker')}
                  >
                    Thinker
                    {thinkerData?.is_active && <span className="thinker-active-dot" />}
                  </button>
                </div>
              )}
              <div className={`status-indicator ${status}`}>
                <div className="status-dot"></div>
                {status}
              </div>
            </div>
          </div>
          
          {thinkerView === 'thinker' ? (
            <div className="thinker-feed" ref={thinkerFeedRef}>
              {thinkerData && thinkerData.events.length > 0 ? (
                thinkerData.events.map((ev, i) => {
                  if (ev.type === 'text') {
                    return (
                      <div key={i} className="thinker-event thinker-event-text">
                        {ev.content}
                      </div>
                    )
                  }
                  if (ev.type === 'tool_call') {
                    return (
                      <div key={i} className="thinker-event thinker-event-tool">
                        <span className="thinker-tool-badge">{ev.tool}</span>
                        <span className="thinker-summary">{ev.summary}</span>
                      </div>
                    )
                  }
                  if (ev.type === 'tool_result') {
                    return (
                      <div key={i} className="thinker-event thinker-event-result">
                        {ev.tool && <span className="thinker-tool-badge" style={{background: 'rgba(16,185,129,0.1)', color: '#10b981', borderColor: 'rgba(16,185,129,0.25)'}}>{ev.tool}</span>}
                        {ev.summary}
                      </div>
                    )
                  }
                  return null
                })
              ) : (
                <div className="terminal-empty">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeOpacity="0.5"><circle cx="12" cy="12" r="10"></circle><path d="M12 8v4l3 3"></path></svg>
                  <p>{thinkerData ? 'No thinker activity yet.' : 'Loading thinker data...'}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="terminal-body" ref={terminalRef}>
              {logs.length > 0 ? (
                logs.map((log, i) => (
                  <div key={i} className="terminal-line">{log}</div>
                ))
              ) : (
                <div className="terminal-empty">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeOpacity="0.5">
                    <circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline>
                  </svg>
                  <p>System idle. Ready for tasks.</p>
                </div>
              )}
            </div>
          )}
        </div>
        )}
      </main>
    </div>
  )
}
