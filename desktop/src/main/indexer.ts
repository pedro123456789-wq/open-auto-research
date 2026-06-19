/**
 * Scans backend/runs/ on startup and watches for archive.json changes.
 * Keeps the SQLite DB in sync with the filesystem.
 */
import fs from 'fs'
import path from 'path'
import chokidar from 'chokidar'
import { upsertRun, upsertNode, getRun } from './db'
import type { NodeRow } from './db'

export type UpdateCallback = (runDir: string) => void

// Resolved once at startup relative to this file:
//   desktop/src/main/ -> ../../.. -> repo root -> backend/runs
const RUNS_DIR = path.resolve(__dirname, '..', '..', '..', '..', 'backend', 'runs')

export function getRunsDir(): string {
  return RUNS_DIR
}

export function getBackendDir(): string {
  return path.resolve(RUNS_DIR, '..')
}

// ── meta.json ─────────────────────────────────────────────────────────────────

interface MetaJson {
  run_name: string
  started_at: string
}

function readMeta(runDir: string): MetaJson | null {
  const metaPath = path.join(runDir, 'meta.json')
  if (!fs.existsSync(metaPath)) return null
  try {
    return JSON.parse(fs.readFileSync(metaPath, 'utf8'))
  } catch {
    return null
  }
}

// ── archive.json ──────────────────────────────────────────────────────────────

interface ArchiveNode {
  node_id: string
  parent_id: string | null
  round: number
  accuracy: number
  compiles: boolean
  compile_error: string | null
  reasoning: string
  novelty: string
  model: string
  timestamp: string
  metadata?: { correct?: number; total?: number }
  correct?: number
  total?: number
}

function ingestArchive(runDir: string): void {
  const archivePath = path.join(runDir, 'archive.json')
  if (!fs.existsSync(archivePath)) return

  let nodes: ArchiveNode[]
  try {
    nodes = JSON.parse(fs.readFileSync(archivePath, 'utf8'))
  } catch {
    return
  }

  const meta = readMeta(runDir)
  const run_name = meta?.run_name ?? path.basename(runDir)
  const started_at = meta?.started_at ?? ''
  const existing = getRun(runDir)

  const best_accuracy = nodes.reduce((m, n) => Math.max(m, n.accuracy ?? 0), 0)

  upsertRun(
    runDir,
    run_name,
    existing?.status ?? 'done',
    existing?.pid ?? null,
    started_at,
    best_accuracy,
    nodes.length
  )

  for (const n of nodes) {
    const row: NodeRow = {
      run_dir: runDir,
      node_id: n.node_id,
      parent_id: n.parent_id ?? null,
      round: n.round ?? 0,
      accuracy: n.accuracy ?? 0,
      correct: n.metadata?.correct ?? n.correct ?? 0,
      total: n.metadata?.total ?? n.total ?? 0,
      compiles: n.compiles ? 1 : 0,
      model: n.model ?? '',
      timestamp: n.timestamp ?? '',
      reasoning: n.reasoning ?? '',
      novelty: n.novelty ?? '',
    }
    upsertNode(row)
  }
}

// ── initial scan ──────────────────────────────────────────────────────────────

export function scanExistingRuns(): void {
  if (!fs.existsSync(RUNS_DIR)) return
  for (const entry of fs.readdirSync(RUNS_DIR, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue
    ingestArchive(path.join(RUNS_DIR, entry.name))
  }
}

// ── live watcher ──────────────────────────────────────────────────────────────

export function watchRuns(onUpdate: UpdateCallback): () => void {
  const watcher = chokidar.watch(path.join(RUNS_DIR, '*', 'archive.json'), {
    ignoreInitial: true,
    awaitWriteFinish: { stabilityThreshold: 300, pollInterval: 100 },
  })

  watcher.on('add', (filePath: string) => {
    const runDir = path.dirname(filePath)
    ingestArchive(runDir)
    onUpdate(runDir)
  })

  watcher.on('change', (filePath: string) => {
    const runDir = path.dirname(filePath)
    ingestArchive(runDir)
    onUpdate(runDir)
  })

  return () => watcher.close()
}
