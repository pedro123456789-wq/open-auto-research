import { useMemo } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { NodeRow } from '../../../shared/types'

interface Props {
  nodes: NodeRow[]
  selectedNodeId: string | null
  onSelect(nodeId: string): void
}

function accuracyColor(acc: number): string {
  if (acc >= 0.6) return '#4ade80'
  if (acc >= 0.4) return '#facc15'
  return '#f87171'
}

const NODE_W = 140
const NODE_H = 56
const H_GAP = 60
const V_GAP = 80

/** Lay out nodes top-to-bottom by round, left-to-right by insertion order. */
function layoutNodes(rows: NodeRow[]): { x: number; y: number; id: string }[] {
  const byRound = new Map<number, NodeRow[]>()
  for (const n of rows) {
    const r = n.round ?? 0
    ;(byRound.get(r) ?? byRound.set(r, []).get(r)!).push(n)
  }
  const rounds = [...byRound.keys()].sort((a, b) => a - b)
  const positions: { x: number; y: number; id: string }[] = []
  for (const round of rounds) {
    const group = byRound.get(round)!
    const totalW = group.length * NODE_W + (group.length - 1) * H_GAP
    const startX = -totalW / 2
    group.forEach((n, i) => {
      positions.push({ id: n.node_id, x: startX + i * (NODE_W + H_GAP), y: round * (NODE_H + V_GAP) })
    })
  }
  return positions
}

export default function NodeTree({ nodes, selectedNodeId, onSelect }: Props) {
  const { flowNodes, flowEdges } = useMemo(() => {
    const positions = layoutNodes(nodes)
    const posMap = new Map(positions.map((p) => [p.id, p]))

    const flowNodes: Node[] = nodes.map((n) => {
      const pos = posMap.get(n.node_id) ?? { x: 0, y: 0 }
      const color = accuracyColor(n.accuracy)
      const isSelected = n.node_id === selectedNodeId
      return {
        id: n.node_id,
        position: { x: pos.x, y: pos.y },
        data: { label: `${n.node_id}\n${(n.accuracy * 100).toFixed(1)}%` },
        style: {
          background: isSelected ? '#7c6af7' : '#1a1a22',
          border: `2px solid ${isSelected ? '#a89cf7' : color}`,
          borderRadius: 8,
          color: '#e4e4ef',
          fontSize: 11,
          width: NODE_W,
          height: NODE_H,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center' as const,
          whiteSpace: 'pre-line' as const,
          cursor: 'pointer',
          fontWeight: isSelected ? 700 : 400,
        },
      }
    })

    const flowEdges: Edge[] = nodes
      .filter((n) => n.parent_id)
      .map((n) => ({
        id: `${n.parent_id}->${n.node_id}`,
        source: n.parent_id!,
        target: n.node_id,
        style: { stroke: '#2e2e3e', strokeWidth: 1.5 },
        animated: n.node_id === selectedNodeId,
      }))

    return { flowNodes, flowEdges }
  }, [nodes, selectedNodeId])

  if (nodes.length === 0) {
    return <div className="empty" style={{ height: '100%' }}>No nodes yet</div>
  }

  return (
    <div className="tree-panel">
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        onNodeClick={(_e, node) => onSelect(node.id)}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#2e2e3e" gap={24} />
        <Controls style={{ background: '#1a1a22', border: '1px solid #2e2e3e' }} />
        <MiniMap
          nodeColor={(n) => (n.style?.border as string)?.split(' ')[2] ?? '#444'}
          style={{ background: '#1a1a22', border: '1px solid #2e2e3e' }}
          maskColor="#0f0f1388"
        />
      </ReactFlow>
    </div>
  )
}
