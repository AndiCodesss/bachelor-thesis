import { useEffect, useState } from 'react'

import type { CacheConfig } from '../types'

export function useCacheConfig(apiUrl: string) {
  const [cacheConfig, setCacheConfig] = useState<CacheConfig | null>(null)

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const resp = await fetch(`${apiUrl}/config/cache`)
        const data = await resp.json()
        if (!cancelled) {
          setCacheConfig(data)
        }
      } catch (error) {
        console.error('Failed to fetch cache config:', error)
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [apiUrl])

  return cacheConfig
}
