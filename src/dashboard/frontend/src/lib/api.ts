const defaultApiOrigin = 'http://localhost:8000'

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, '')
}

const apiOrigin = trimTrailingSlash(import.meta.env.VITE_API_ORIGIN ?? defaultApiOrigin)
const wsOrigin = trimTrailingSlash(import.meta.env.VITE_WS_ORIGIN ?? apiOrigin.replace(/^http/i, 'ws'))

export const API_URL = `${apiOrigin}/api`
export const WS_URL = `${wsOrigin}/ws`
