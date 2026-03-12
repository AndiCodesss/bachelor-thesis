import { useEffect, useState } from 'react'

import type { AutonomyStatus } from '../types'

interface UseAutonomyMonitorOptions {
  apiUrl: string
  enabled: boolean
}

export function useAutonomyMonitor({ apiUrl, enabled }: UseAutonomyMonitorOptions) {
  const [statusData, setStatusData] = useState<AutonomyStatus | null>(null)

  useEffect(() => {
    if (!enabled) {
      return
    }

    let cancelled = false

    const load = async () => {
      try {
        const statusResp = await fetch(`${apiUrl}/autonomy/status`)
        const statusJson = await statusResp.json()

        if (!cancelled) {
          setStatusData(statusJson)
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

  return { statusData }
}
