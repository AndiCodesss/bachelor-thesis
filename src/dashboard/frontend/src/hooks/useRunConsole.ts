import { useEffect, useRef, useState } from 'react'

import type { RunStatus } from '../types'

interface UseRunConsoleOptions {
  apiUrl: string
  wsUrl: string
}

export function useRunConsole({ apiUrl, wsUrl }: UseRunConsoleOptions) {
  const [runId, setRunId] = useState<string | null>(null)
  const [status, setStatus] = useState<RunStatus>('idle')
  const [logs, setLogs] = useState<string[]>([])
  const [currentCmd, setCurrentCmd] = useState('')

  const wsRef = useRef<WebSocket | null>(null)

  const syncRunStatus = async (id: string) => {
    try {
      const resp = await fetch(`${apiUrl}/runs/${id}`)
      const data = await resp.json()
      if (typeof data.status === 'string') {
        setStatus(data.status as RunStatus)
      }
    } catch (error) {
      console.error('Failed to refresh run status:', error)
    }
  }

  const appendLog = (line: string) => {
    setLogs(prev => [...prev, line])
  }

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const connectWebSocket = (id: string, cmdStr: string) => {
    setRunId(id)
    setCurrentCmd(cmdStr)
    if (wsRef.current) {
      wsRef.current.close()
    }

    const ws = new WebSocket(`${wsUrl}/runs/${id}/logs`)
    wsRef.current = ws
    ws.onmessage = event => {
      appendLog(event.data)
    }
    ws.onclose = () => {
      void syncRunStatus(id)
    }
  }

  const triggerRun = async (endpoint: string, payload: Record<string, unknown>) => {
    if (status === 'running') {
      return
    }
    setLogs([])
    setStatus('running')

    try {
      const resp = await fetch(`${apiUrl}/run/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await resp.json()
      if (!resp.ok) {
        throw new Error(String(data.detail ?? data.error ?? `Request failed with status ${resp.status}`))
      }
      if (!data.run_id || !data.cmd) {
        throw new Error('Backend response missing run metadata.')
      }
      connectWebSocket(data.run_id, data.cmd)
    } catch (error) {
      console.error(error)
      setStatus('failed')
      setLogs([error instanceof Error ? error.message : 'Failed to connect to backend server.'])
    }
  }

  const stopRun = async () => {
    if (!runId) {
      return
    }
    try {
      const resp = await fetch(`${apiUrl}/runs/${runId}/stop`, { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok || data.error) {
        throw new Error(String(data.error ?? `Request failed with status ${resp.status}`))
      }
      if (typeof data.status === 'string' && data.status === 'stopped') {
        setStatus('failed')
      }
      if (typeof data.status === 'string' && ['stopped', 'already_stopped'].includes(data.status)) {
        setRunId(null)
        setCurrentCmd('')
      }
      await syncRunStatus(runId)
    } catch (error) {
      console.error('Failed to stop run:', error)
      appendLog(error instanceof Error ? error.message : 'Failed to stop run.')
    }
  }

  return {
    canStop: runId !== null && status !== 'idle',
    currentCmd,
    logs,
    runId,
    status,
    stopRun,
    triggerRun,
  }
}
