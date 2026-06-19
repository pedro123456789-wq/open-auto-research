import type { RunRow } from '../../../shared/types'
import StatusBadge from './StatusBadge'

interface Props {
  runs: RunRow[]
  selectedDir: string | null
  onSelect(runDir: string): void
  anyRunning: boolean
  onStartClick(): void
}

export default function Sidebar({ runs, selectedDir, onSelect, anyRunning, onStartClick }: Props) {
  // Group by run_name
  const groups = runs.reduce<Record<string, RunRow[]>>((acc, r) => {
    ;(acc[r.run_name] ??= []).push(r)
    return acc
  }, {})

  return (
    <aside className="sidebar">
      <div className="sidebar-title">Runs</div>
      <div className="sidebar-runs">
        {Object.keys(groups).length === 0 && (
          <div style={{ padding: '12px 14px', color: 'var(--text-muted)', fontSize: 12 }}>
            No runs yet
          </div>
        )}
        {Object.entries(groups).map(([name, groupRuns]) => (
          <div key={name}>
            <div className="sidebar-group-label">{name}</div>
            {groupRuns.map((run) => {
              const ts = run.started_at ? new Date(run.started_at).toLocaleString() : '—'
              const acc = run.best_accuracy > 0 ? `${(run.best_accuracy * 100).toFixed(1)}%` : '—'
              return (
                <div
                  key={run.run_dir}
                  className={`sidebar-item${run.run_dir === selectedDir ? ' active' : ''}`}
                  onClick={() => onSelect(run.run_dir)}
                >
                  <div className="sidebar-item-label">
                    <StatusBadge status={run.status} />
                  </div>
                  <div className="sidebar-item-meta">
                    {ts} · {run.node_count} nodes · best {acc}
                  </div>
                </div>
              )
            })}
          </div>
        ))}
      </div>
      <div className="sidebar-footer">
        <button
          className="btn btn-primary btn-full"
          onClick={onStartClick}
          disabled={anyRunning}
          title={anyRunning ? 'A run is already active' : 'Start a new run'}
        >
          {anyRunning ? '● Running…' : '+ New Run'}
        </button>
      </div>
    </aside>
  )
}
