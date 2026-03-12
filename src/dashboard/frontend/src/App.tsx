import { useEffect, useState } from 'react'

import { TerminalPanel } from './components/TerminalPanel'
import { AutonomyTab } from './components/tabs/AutonomyTab'
import { CacheTab } from './components/tabs/CacheTab'
import { CleanupTab } from './components/tabs/CleanupTab'
import { SignalsTab } from './components/tabs/SignalsTab'
import { useAutonomyMonitor } from './hooks/useAutonomyMonitor'
import { useCacheConfig } from './hooks/useCacheConfig'
import { useRunConsole } from './hooks/useRunConsole'
import { useSignalsExplorer } from './hooks/useSignalsExplorer'
import { API_URL, WS_URL } from './lib/api'
import type { TabType } from './types'

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('cache')

  const cacheConfig = useCacheConfig(API_URL)
  const {
    currentCmd,
    logs,
    status,
    stopRun,
    triggerRun,
  } = useRunConsole({ apiUrl: API_URL, wsUrl: WS_URL })
  const { statusData, thinkerData } = useAutonomyMonitor({
    apiUrl: API_URL,
    enabled: activeTab === 'autonomy',
  })
  const {
    selectedSignal,
    setSelectedSignal,
    signalDetails,
    signalsList,
  } = useSignalsExplorer({
    apiUrl: API_URL,
    enabled: activeTab === 'signals',
  })

  const [cacheSplits, setCacheSplits] = useState<string[]>(['all'])
  const [cacheSession, setCacheSession] = useState('eth')
  const [cacheExecMode, setCacheExecMode] = useState('auto')
  const [cacheBarFilter, setCacheBarFilter] = useState('')
  const [cacheClean, setCacheClean] = useState(false)

  const [missionPath, setMissionPath] = useState('configs/missions/alpha-discovery.yaml')
  const [agentConfigPath, setAgentConfigPath] = useState('configs/agents/llm_orchestrator.yaml')
  const [freshState, setFreshState] = useState(false)
  const [allowBootstrap, setAllowBootstrap] = useState(false)
  const [useNotebookLM, setUseNotebookLM] = useState(true)
  const [validatorOnly, setValidatorOnly] = useState(false)
  const [orchestratorOnly, setOrchestratorOnly] = useState(false)
  const [laneCount, setLaneCount] = useState(2)
  const [thinkerView, setThinkerView] = useState<'logs' | 'thinker'>('logs')

  const [cleanupForce, setCleanupForce] = useState(false)

  useEffect(() => {
    if (!cacheConfig) {
      return
    }
    if (cacheConfig.session_filters.length > 0 && !cacheConfig.session_filters.includes(cacheSession)) {
      setCacheSession(cacheConfig.session_filters[0])
    }
    if (cacheConfig.exec_modes.length > 0 && !cacheConfig.exec_modes.includes(cacheExecMode)) {
      setCacheExecMode(cacheConfig.exec_modes[0])
    }
  }, [cacheConfig, cacheExecMode, cacheSession])

  const startCache = () => {
    void triggerRun('cache', {
      splits: cacheSplits,
      session_filter: cacheSession,
      exec_mode: cacheExecMode,
      bar_filter: cacheBarFilter || null,
      clean: cacheClean,
    })
  }

  const startAutonomy = () => {
    void triggerRun('autonomy', {
      mission: missionPath,
      agent_config: agentConfigPath,
      no_resume: freshState,
      allow_bootstrap: allowBootstrap,
      use_notebooklm: useNotebookLM,
      validator_only: validatorOnly,
      orchestrator_only: orchestratorOnly,
      lane_count: laneCount,
    })
  }

  const startCleanup = () => {
    void triggerRun('cleanup', { force: cleanupForce })
  }

  return (
    <div className="container">
      <header>
        <div className="header-left">
          <h1>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
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
        <div className="panel config-panel" style={activeTab === 'signals' ? { flex: 1, maxWidth: 'none', overflow: 'hidden' } : {}}>
          {activeTab === 'cache' && (
            <CacheTab
              cacheConfig={cacheConfig}
              splits={cacheSplits}
              session={cacheSession}
              execMode={cacheExecMode}
              barFilter={cacheBarFilter}
              clean={cacheClean}
              status={status}
              onSplitsChange={setCacheSplits}
              onSessionChange={setCacheSession}
              onExecModeChange={setCacheExecMode}
              onBarFilterChange={setCacheBarFilter}
              onCleanChange={setCacheClean}
              onStart={startCache}
              onStop={stopRun}
            />
          )}

          {activeTab === 'autonomy' && (
            <AutonomyTab
              agentConfig={agentConfigPath}
              allowBootstrap={allowBootstrap}
              laneCount={laneCount}
              mission={missionPath}
              noResume={freshState}
              orchestratorOnly={orchestratorOnly}
              status={status}
              statusData={statusData}
              useNotebookLM={useNotebookLM}
              validatorOnly={validatorOnly}
              onAgentConfigChange={setAgentConfigPath}
              onAllowBootstrapChange={setAllowBootstrap}
              onLaneCountChange={setLaneCount}
              onMissionChange={setMissionPath}
              onNoResumeChange={setFreshState}
              onOrchestratorOnlyChange={setOrchestratorOnly}
              onStart={startAutonomy}
              onStop={stopRun}
              onUseNotebookLMChange={setUseNotebookLM}
              onValidatorOnlyChange={setValidatorOnly}
            />
          )}

          {activeTab === 'cleanup' && (
            <CleanupTab
              force={cleanupForce}
              status={status}
              onForceChange={setCleanupForce}
              onStart={startCleanup}
              onStop={stopRun}
            />
          )}

          {activeTab === 'signals' && (
            <SignalsTab
              selectedSignal={selectedSignal}
              signalDetails={signalDetails}
              signalsList={signalsList}
              onSelectSignal={setSelectedSignal}
            />
          )}
        </div>

        {activeTab !== 'signals' && (
          <TerminalPanel
            activeTab={activeTab}
            currentCmd={currentCmd}
            logs={logs}
            status={status}
            thinkerData={thinkerData}
            thinkerView={thinkerView}
            onThinkerViewChange={setThinkerView}
          />
        )}
      </main>
    </div>
  )
}
