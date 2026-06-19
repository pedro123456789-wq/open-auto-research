import { spawn, ChildProcess } from 'child_process'
import fs from 'fs'
import path from 'path'
import { upsertRun, setRunStatus } from './db'
import { getRunsDir, getBackendDir } from './indexer'

let activeChild: ChildProcess | null = null
let activeRunDir: string | null = null

export function isRunning(): boolean {
  return activeChild !== null
}

export function getActiveRunDir(): string | null {
  return activeRunDir
}

// ── validation ────────────────────────────────────────────────────────────────

export function validateRunName(runName: string): string | null {
  const backendDir = getBackendDir()
  const required = ['baselines', 'evaluators', 'improvement_agents']
  for (const folder of required) {
    const dir = path.join(backendDir, folder, runName)
    if (!fs.existsSync(dir)) {
      return `Missing folder: backend/${folder}/${runName}`
    }
  }
  return null
}

// ── launch ────────────────────────────────────────────────────────────────────

export interface StartResult {
  ok: boolean
  error?: string
  run_dir?: string
}

export function startRun(runName: string, onUpdate: (runDir: string) => void): StartResult {
  if (activeChild) {
    return { ok: false, error: 'A run is already active.' }
  }

  const validationError = validateRunName(runName)
  if (validationError) {
    return { ok: false, error: validationError }
  }

  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19).replace('T', '_')
  const runDir = path.join(getRunsDir(), ts)
  fs.mkdirSync(path.join(runDir, 'nodes'), { recursive: true })

  const startedAt = new Date().toISOString()
  fs.writeFileSync(
    path.join(runDir, 'meta.json'),
    JSON.stringify({ run_name: runName, started_at: startedAt }, null, 2)
  )

  upsertRun(runDir, runName, 'running', null, startedAt, 0, 0)

  const child = spawn(
    'python3',
    ['run_self_improvement.py', '--run-name', runName, '--run-dir', runDir],
    {
      cwd: getBackendDir(),
      detached: true,
      stdio: 'ignore',
    }
  )

  child.unref()

  upsertRun(runDir, runName, 'running', child.pid ?? null, startedAt, 0, 0)

  activeChild = child
  activeRunDir = runDir

  child.on('exit', (code) => {
    const status = code === 0 ? 'done' : 'failed'
    setRunStatus(runDir, status, new Date().toISOString())
    activeChild = null
    activeRunDir = null
    onUpdate(runDir)
  })

  return { ok: true, run_dir: runDir }
}

// ── stop ──────────────────────────────────────────────────────────────────────

export function stopRun(): void {
  if (!activeChild || activeChild.pid == null) return
  try {
    process.kill(-activeChild.pid, 'SIGTERM')
  } catch {
    try { activeChild.kill('SIGTERM') } catch { /* already gone */ }
  }
  if (activeRunDir) {
    setRunStatus(activeRunDir, 'interrupted', new Date().toISOString())
  }
  activeChild = null
  activeRunDir = null
}

export function killActiveOnQuit(): void {
  if (!activeChild || activeChild.pid == null) return
  try {
    process.kill(-activeChild.pid, 'SIGKILL')
  } catch {
    try { activeChild.kill('SIGKILL') } catch { /* already gone */ }
  }
  if (activeRunDir) {
    setRunStatus(activeRunDir, 'interrupted', new Date().toISOString())
  }
}
