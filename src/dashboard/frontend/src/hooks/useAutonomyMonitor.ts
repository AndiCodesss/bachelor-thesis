import { useEffect, useState } from 'react'

import type { AutonomyStatus, ThinkerData } from '../types'

interface UseAutonomyMonitorOptions {
  apiUrl: string
  enabled: boolean
}

export function useAutonomyMonitor({ apiUrl, enabled }: UseAutonomyMonitorOptions) {
  const [statusData, setStatusData] = useState<AutonomyStatus | null>(null)
  const [thinkerData, setThinkerData] = useState<ThinkerData | null>(null)

  useEffect(() => {
    if (!enabled) {
      return
    }

    let cancelled = false

    const load = async () => {
      try {
        const [statusResp, thinkerResp] = await Promise.all([
          fetch(`${apiUrl}/autonomy/status`),
          fetch(`${apiUrl}/autonomy/thinker`),
        ])
        const [statusJson, thinkerJson] = await Promise.all([
          statusResp.json(),
          thinkerResp.json(),
        ])

        if (!cancelled) {
          setStatusData(statusJson)
          setThinkerData(thinkerJson)
        }
      } catch (error) {
        console.error('Failed to fetch autonomy monitor data:', error)
      }
    }

    void load()
    const interval = window.setInterval(() => {
      void load()
    }, 3000)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [apiUrl, enabled])

  return { statusData, thinkerData }
}
