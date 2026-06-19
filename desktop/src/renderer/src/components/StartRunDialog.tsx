import { useState, useRef, useEffect } from 'react'

interface Props {
  onClose(): void
  onStarted(runDir: string): void
}

export default function StartRunDialog({ onClose, onStarted }: Props) {
  const [runName, setRunName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { inputRef.current?.focus() }, [])

  async function handleStart() {
    const name = runName.trim()
    if (!name) { setError('Run name is required.'); return }
    setError('')
    setLoading(true)
    try {
      const result = await window.api.startRun(name)
      if (result.ok && result.run_dir) {
        onStarted(result.run_dir)
      } else {
        setError(result.error ?? 'Failed to start run.')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter') handleStart()
    if (e.key === 'Escape') onClose()
  }

  return (
    <div className="dialog-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog">
        <h2>Start New Run</h2>
        <label>Run name</label>
        <input
          ref={inputRef}
          value={runName}
          onChange={(e) => setRunName(e.target.value)}
          onKeyDown={handleKey}
          placeholder="e.g. agentic_mem"
          disabled={loading}
        />
        <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
          Must match folders in backend/baselines/, backend/evaluators/, backend/improvement_agents/
        </div>
        {error && <div className="dialog-error">{error}</div>}
        <div className="dialog-actions">
          <button className="btn" onClick={onClose} disabled={loading}>Cancel</button>
          <button className="btn btn-primary" onClick={handleStart} disabled={loading || !runName.trim()}>
            {loading ? 'Starting…' : 'Start'}
          </button>
        </div>
      </div>
    </div>
  )
}
