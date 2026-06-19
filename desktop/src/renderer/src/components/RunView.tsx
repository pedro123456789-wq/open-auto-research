import { useState } from 'react'
import type { RunRow, NodeRow } from '../../../shared/types'
import StatusBadge from './StatusBadge'
import NodeTree from './NodeTree'
import NodeDetail from './NodeDetail'

interface Props {
  run: RunRow
  nodes: NodeRow[]
}

export default function RunView({ run, nodes }: Props) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const selectedNode = nodes.find((n) => n.node_id === selectedNodeId) ?? null
  const bestAcc = run.best_accuracy > 0 ? `${(run.best_accuracy * 100).toFixed(1)}%` : '—'

  return (
    <div className="run-view">
      <div className="run-header">
        <div>
          <div className="run-title">{run.run_name}</div>
          <div style={{ marginTop: 4 }}>
            <StatusBadge status={run.status} />
          </div>
        </div>
        <div className="run-meta">
          <div className="run-meta-item">
            <div className="run-meta-label">Best accuracy</div>
            <div className="run-meta-value" style={{ color: 'var(--accent)' }}>{bestAcc}</div>
          </div>
          <div className="run-meta-item">
            <div className="run-meta-label">Nodes</div>
            <div className="run-meta-value">{run.node_count}</div>
          </div>
          <div className="run-meta-item">
            <div className="run-meta-label">Started</div>
            <div className="run-meta-value" style={{ fontSize: 12 }}>
              {run.started_at ? new Date(run.started_at).toLocaleString() : '—'}
            </div>
          </div>
        </div>
      </div>
      <div className="run-body">
        <NodeTree
          nodes={nodes}
          selectedNodeId={selectedNodeId}
          onSelect={setSelectedNodeId}
        />
        <NodeDetail node={selectedNode} />
      </div>
    </div>
  )
}
