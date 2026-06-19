type Status = 'running' | 'done' | 'failed' | 'interrupted' | string

const LABELS: Record<string, string> = {
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
  interrupted: 'Interrupted',
}

export default function StatusBadge({ status }: { status: Status }) {
  return (
    <span className={`badge badge-${status}`}>
      {LABELS[status] ?? status}
    </span>
  )
}
