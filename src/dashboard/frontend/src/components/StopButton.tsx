interface StopButtonProps {
  onClick: () => void
}

export function StopButton({ onClick }: StopButtonProps) {
  return (
    <button className="action-btn danger" onClick={onClick} style={{ padding: '0.8rem 1.5rem' }}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      </svg>
    </button>
  )
}
