export function formatMetricKey(key: string): string {
  return key
    .split('_')
    .map(word => (word ? word.charAt(0).toUpperCase() + word.slice(1) : ''))
    .join(' ')
}

export function formatMetricValue(key: string, value: unknown): string {
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

export function formatAutonomyTimestamp(value: string): string {
  if (!value) {
    return 'unknown time'
  }
  const dt = new Date(value)
  if (Number.isNaN(dt.getTime())) {
    return value
  }
  return dt.toLocaleString([], {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatUsd(value: number | null | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'n/a'
  }
  const sign = value > 0 ? '+' : ''
  return `${sign}$${value.toFixed(2)}`
}
