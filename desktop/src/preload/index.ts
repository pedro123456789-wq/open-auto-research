import { contextBridge, ipcRenderer } from 'electron'
import type { RunRow, NodeRow } from '../shared/types'

contextBridge.exposeInMainWorld('api', {
  listRuns: (): Promise<RunRow[]> =>
    ipcRenderer.invoke('runs:list'),

  getRun: (runDir: string): Promise<RunRow | null> =>
    ipcRenderer.invoke('runs:get', runDir),

  getNodes: (runDir: string): Promise<NodeRow[]> =>
    ipcRenderer.invoke('nodes:get', runDir),

  isRunning: (): Promise<boolean> =>
    ipcRenderer.invoke('runs:isRunning'),

  startRun: (runName: string): Promise<{ ok: boolean; error?: string; run_dir?: string }> =>
    ipcRenderer.invoke('runs:start', runName),

  stopRun: (): Promise<void> =>
    ipcRenderer.invoke('runs:stop'),

  onRunUpdated: (cb: (run: RunRow) => void) => {
    ipcRenderer.on('run:updated', (_e, run) => cb(run))
  },

  onListChanged: (cb: () => void) => {
    ipcRenderer.on('runs:listChanged', () => cb())
  },

  removeAllListeners: () => {
    ipcRenderer.removeAllListeners('run:updated')
    ipcRenderer.removeAllListeners('runs:listChanged')
  },
})
