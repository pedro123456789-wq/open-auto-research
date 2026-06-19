import type { NodeRow } from '../../../shared/types'

export default function NodeDetail({ node }: { node: NodeRow | null }) {
  if (!node) {
    return (
      <div className="detail-panel">
        <div className="empty" style={{ height: '100%' }}>Select a node</div>
      </div>
    )
  }

  return (
    <div className="detail-panel">
      <div className="detail-section">
        <div className="detail-accuracy">{(node.accuracy * 100).toFixed(1)}%</div>
        <div style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 2 }}>
          {node.correct}/{node.total} correct
        </div>
      </div>

      <div className="detail-section">
        <div className="detail-label">Node</div>
        <div className="detail-value" style={{ fontFamily: 'monospace' }}>{node.node_id}</div>
      </div>

      {node.parent_id && (
        <div className="detail-section">
          <div className="detail-label">Parent</div>
          <div className="detail-value" style={{ fontFamily: 'monospace' }}>{node.parent_id}</div>
        </div>
      )}

      <div className="detail-section">
        <div className="detail-label">Round</div>
        <div className="detail-value">{node.round === 0 ? 'Root (seed)' : `Round ${node.round}`}</div>
      </div>

      {node.model && (
        <div className="detail-section">
          <div className="detail-label">Model</div>
          <div className="detail-value">{node.model}</div>
        </div>
      )}

      {node.reasoning && (
        <div className="detail-section">
          <div className="detail-label">Reasoning</div>
          <div className="detail-value" style={{ whiteSpace: 'pre-wrap', fontSize: 12, color: 'var(--text-muted)' }}>
            {node.reasoning.slice(0, 600)}{node.reasoning.length > 600 ? '…' : ''}
          </div>
        </div>
      )}

      {node.novelty && (
        <div className="detail-section">
          <div className="detail-label">Novelty</div>
          <div className="detail-value" style={{ whiteSpace: 'pre-wrap', fontSize: 12, color: 'var(--text-muted)' }}>
            {node.novelty.slice(0, 400)}{node.novelty.length > 400 ? '…' : ''}
          </div>
        </div>
      )}
    </div>
  )
}
