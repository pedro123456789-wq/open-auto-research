export interface RunRow {
  run_dir: string
  run_name: string
  status: string
  pid: number | null
  started_at: string | null
  finished_at: string | null
  best_accuracy: number
  node_count: number
}

export interface NodeRow {
  run_dir: string
  node_id: string
  parent_id: string | null
  round: number
  accuracy: number
  correct: number
  total: number
  compiles: number
  model: string
  timestamp: string
  reasoning: string
  novelty: string
}

export interface StartResult {
  ok: boolean
  error?: string
  run_dir?: string
}
