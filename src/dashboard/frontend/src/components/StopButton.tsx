interface StopButtonProps {
  onClick: () => void
  label?: string
}

export function StopButton({ onClick, label = 'Stop current run' }: StopButtonProps) {
  return (
    <button
      type="button"
      className="action-btn action-btn--icon danger"
      onClick={onClick}
      aria-label={label}
      title={label}
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      </svg>
    </button>
  )
}
