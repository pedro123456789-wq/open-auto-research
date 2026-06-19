import Database from 'better-sqlite3'
import path from 'path'
import { app } from 'electron'
import type { RunRow, NodeRow } from '../shared/types'

let db: Database.Database

export function openDb(): Database.Database {
  if (db) return db
  const dbPath = path.join(app.getPath('userData'), 'runs.sqlite')
  db = new Database(dbPath)
  db.pragma('journal_mode = WAL')
  db.pragma('foreign_keys = ON')

  db.exec(`
    CREATE TABLE IF NOT EXISTS runs (
      run_dir       TEXT PRIMARY KEY,
      run_name      TEXT NOT NULL,
      status        TEXT NOT NULL DEFAULT 'running',
      pid           INTEGER,
      started_at    TEXT,
      finished_at   TEXT,
      best_accuracy REAL DEFAULT 0,
      node_count    INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS nodes (
      run_dir   TEXT NOT NULL,
      node_id   TEXT NOT NULL,
      parent_id TEXT,
      round     INTEGER,
      accuracy  REAL,
      correct   INTEGER,
      total     INTEGER,
      compiles  INTEGER,
      model     TEXT,
      timestamp TEXT,
      reasoning TEXT,
      novelty   TEXT,
      PRIMARY KEY (run_dir, node_id),
      FOREIGN KEY (run_dir) REFERENCES runs(run_dir) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(run_name);
    CREATE INDEX IF NOT EXISTS idx_nodes_run ON nodes(run_dir);
  `)

  return db
}

export function closeDb(): void {
  db?.close()
}

// ── runs ─────────────────────────────────────────────────────────────────────

export function upsertRun(
  run_dir: string,
  run_name: string,
  status: string,
  pid: number | null,
  started_at: string,
  best_accuracy: number,
  node_count: number
): void {
  openDb().prepare(`
    INSERT INTO runs (run_dir, run_name, status, pid, started_at, best_accuracy, node_count)
    VALUES (@run_dir, @run_name, @status, @pid, @started_at, @best_accuracy, @node_count)
    ON CONFLICT(run_dir) DO UPDATE SET
      status = excluded.status,
      pid = excluded.pid,
      best_accuracy = excluded.best_accuracy,
      node_count = excluded.node_count
  `).run({ run_dir, run_name, status, pid, started_at, best_accuracy, node_count })
}

export function setRunStatus(
  run_dir: string,
  status: string,
  finished_at: string | null = null
): void {
  openDb().prepare(`
    UPDATE runs SET status = ?, finished_at = ? WHERE run_dir = ?
  `).run(status, finished_at, run_dir)
}

export function markStaleRunsInterrupted(): void {
  openDb().prepare(`
    UPDATE runs SET status = 'interrupted', finished_at = datetime('now')
    WHERE status = 'running'
  `).run()
}

export function listRuns(): RunRow[] {
  return openDb().prepare(`
    SELECT * FROM runs ORDER BY run_name ASC, started_at DESC
  `).all() as RunRow[]
}

export function getRun(run_dir: string): RunRow | null {
  return (openDb().prepare(`SELECT * FROM runs WHERE run_dir = ?`).get(run_dir) as RunRow) ?? null
}

// ── nodes ─────────────────────────────────────────────────────────────────────

export function upsertNode(n: NodeRow): void {
  openDb().prepare(`
    INSERT INTO nodes
      (run_dir, node_id, parent_id, round, accuracy, correct, total, compiles, model, timestamp, reasoning, novelty)
    VALUES
      (@run_dir, @node_id, @parent_id, @round, @accuracy, @correct, @total, @compiles, @model, @timestamp, @reasoning, @novelty)
    ON CONFLICT(run_dir, node_id) DO UPDATE SET
      accuracy = excluded.accuracy,
      correct  = excluded.correct,
      total    = excluded.total,
      compiles = excluded.compiles,
      reasoning = excluded.reasoning,
      novelty  = excluded.novelty
  `).run(n)
}

export function getNodes(run_dir: string): NodeRow[] {
  return openDb().prepare(`
    SELECT * FROM nodes WHERE run_dir = ? ORDER BY round ASC, timestamp ASC
  `).all(run_dir) as NodeRow[]
}

// ── types ─────────────────────────────────────────────────────────────────────

export type { RunRow, NodeRow } from '../shared/types'
