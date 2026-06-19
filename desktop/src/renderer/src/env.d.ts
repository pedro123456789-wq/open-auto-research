/// <reference types="vite/client" />

import type { RunRow, NodeRow, StartResult } from '../../shared/types'

declare global {
  interface Window {
    api: {
      listRuns(): Promise<RunRow[]>
      getRun(runDir: string): Promise<RunRow | null>
      getNodes(runDir: string): Promise<NodeRow[]>
      isRunning(): Promise<boolean>
      startRun(runName: string): Promise<StartResult>
      stopRun(): Promise<void>
      onRunUpdated(cb: (run: RunRow) => void): void
      onListChanged(cb: () => void): void
      removeAllListeners(): void
    }
  }
}
