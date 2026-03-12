import { useEffect, useState } from 'react'

import type { SignalDetails, SignalItem } from '../types'

interface UseSignalsExplorerOptions {
  apiUrl: string
  enabled: boolean
}

export function useSignalsExplorer({ apiUrl, enabled }: UseSignalsExplorerOptions) {
  const [signalsList, setSignalsList] = useState<SignalItem[]>([])
  const [selectedSignal, setSelectedSignal] = useState<string | null>(null)
  const [signalDetails, setSignalDetails] = useState<SignalDetails | null>(null)

  useEffect(() => {
    if (!enabled) {
      return
    }

    let cancelled = false

    const loadSignals = async () => {
      try {
        const resp = await fetch(`${apiUrl}/signals`)
        const data = await resp.json()
        if (!cancelled) {
          setSignalsList(data)
        }
      } catch (error) {
        console.error('Failed to fetch signals list:', error)
      }
    }

    void loadSignals()
    return () => {
      cancelled = true
    }
  }, [apiUrl, enabled])

  useEffect(() => {
    if (!selectedSignal) {
      return
    }

    let cancelled = false
    setSignalDetails(null)

    const loadSignalDetails = async () => {
      try {
        const resp = await fetch(`${apiUrl}/signals/${selectedSignal}`)
        const data = await resp.json()
        if (!cancelled) {
          setSignalDetails(data)
        }
      } catch (error) {
        console.error('Failed to fetch signal details:', error)
      }
    }

    void loadSignalDetails()
    return () => {
      cancelled = true
    }
  }, [apiUrl, selectedSignal])

  return {
    selectedSignal,
    setSelectedSignal,
    signalDetails,
    signalsList,
  }
}
