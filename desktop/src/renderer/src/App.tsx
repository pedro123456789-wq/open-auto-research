import { useState, useEffect, useCallback } from 'react'
import type { RunRow, NodeRow } from '../../shared/types'
import Sidebar from './components/Sidebar'
import RunView from './components/RunView'
import StartRunDialog from './components/StartRunDialog'

export default function App() {
  const [runs, setRuns] = useState<RunRow[]>([])
  const [selectedDir, setSelectedDir] = useState<string | null>(null)
  const [nodes, setNodes] = useState<NodeRow[]>([])
  const [anyRunning, setAnyRunning] = useState(false)
  const [showDialog, setShowDialog] = useState(false)

  const refreshRuns = useCallback(async () => {
    const all = await window.api.listRuns()
    setRuns(all)
    setAnyRunning(all.some((r) => r.status === 'running'))
  }, [])

  const refreshNodes = useCallback(async (runDir: string) => {
    const n = await window.api.getNodes(runDir)
    setNodes(n)
  }, [])

  useEffect(() => {
    refreshRuns()

    window.api.onListChanged(refreshRuns)

    window.api.onRunUpdated((updated) => {
      setRuns((prev) => prev.map((r) => (r.run_dir === updated.run_dir ? updated : r)))
      setAnyRunning((prev) => {
        // recompute from full list after updating one row
        return updated.status === 'running' ? true : prev
      })
      if (selectedDir === updated.run_dir) {
        refreshNodes(updated.run_dir)
      }
    })

    return () => window.api.removeAllListeners()
  }, [refreshRuns, refreshNodes, selectedDir])

  useEffect(() => {
    if (selectedDir) refreshNodes(selectedDir)
  }, [selectedDir, refreshNodes])

  const selectedRun = runs.find((r) => r.run_dir === selectedDir) ?? null

  return (
    <div className="layout">
      <Sidebar
        runs={runs}
        selectedDir={selectedDir}
        onSelect={setSelectedDir}
        anyRunning={anyRunning}
        onStartClick={() => setShowDialog(true)}
      />
      <main className="main-area">
        {selectedRun ? (
          <RunView run={selectedRun} nodes={nodes} />
        ) : (
          <div className="empty">Select a run from the sidebar</div>
        )}
      </main>
      {showDialog && (
        <StartRunDialog
          onClose={() => setShowDialog(false)}
          onStarted={(runDir) => {
            setShowDialog(false)
            refreshRuns()
            setSelectedDir(runDir)
          }}
        />
      )}
    </div>
  )
}
