export type TabType = 'cache' | 'autonomy' | 'cleanup' | 'signals'
export type RunStatus = 'idle' | 'running' | 'completed' | 'failed'

export interface CacheConfig {
  splits: string[]
  session_filters: string[]
  exec_modes: string[]
  bar_filters: string[]
}

export interface AutonomyStatus {
  queue: { pending: number; in_progress: number; completed: number; failed: number }
  budget: { experiments_run: number; max_experiments: number | 'n/a'; failures: Record<string, number> }
  financial: {
    tested: number
    avg_net_pnl: number
    avg_sharpe: number
    pass_rate_pct: number
    best: { strategy: string; net_pnl: number; sharpe: number; trades: number } | null
    worst: { strategy: string; net_pnl: number; sharpe: number; trades: number } | null
  }
  active_hypotheses: Array<{ id: string; tasks: number }>
  recent_results: Array<{
    strategy: string
    bar: string
    timestamp: string
    verdict: string
    failure_code: string
    signal_count: number | null
    edge_events: number | null
    edge_status: string
    best_horizon_bars: number | null
    best_avg_trade_pnl: number | null
    backtest_trades: number | null
    net_pnl: number | null
    sharpe: number | null
  }>
}

export interface SignalItem {
  strategy: string
  verdict: string
  timestamp: string
  bar_config: string
}

export interface SignalDetails {
  strategy: string
  code: string
  metrics: Record<string, unknown>
  gauntlet: Record<string, unknown>
  verdict: string
  timestamp: string
}

export interface GauntletItem {
  verdict?: string
  msg?: string
}

export interface ThinkerEvent {
  type: 'text' | 'tool_call' | 'tool_result'
  content?: string
  tool?: string
  summary?: string
}

export interface ThinkerData {
  events: ThinkerEvent[]
  session_id: string | null
  is_active: boolean
  last_updated: string | null
}
