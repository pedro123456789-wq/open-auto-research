import { app, BrowserWindow, ipcMain, shell } from 'electron'
import path from 'path'
import { openDb, closeDb, listRuns, getRun, getNodes, markStaleRunsInterrupted } from './db'
import { scanExistingRuns, watchRuns } from './indexer'
import { startRun, stopRun, killActiveOnQuit, isRunning } from './runner'

let mainWindow: BrowserWindow | null = null
let stopWatcher: (() => void) | null = null

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  })

  if (process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'))
  }

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
}

// ── IPC handlers ──────────────────────────────────────────────────────────────

function pushUpdate(runDir: string): void {
  const run = getRun(runDir)
  mainWindow?.webContents.send('run:updated', run)
}

function registerIpc(): void {
  ipcMain.handle('runs:list', () => listRuns())

  ipcMain.handle('runs:get', (_e, runDir: string) => getRun(runDir))

  ipcMain.handle('nodes:get', (_e, runDir: string) => getNodes(runDir))

  ipcMain.handle('runs:isRunning', () => isRunning())

  ipcMain.handle('runs:start', (_e, runName: string) => {
    return startRun(runName, (runDir) => {
      pushUpdate(runDir)
      mainWindow?.webContents.send('runs:listChanged')
    })
  })

  ipcMain.handle('runs:stop', () => {
    stopRun()
    mainWindow?.webContents.send('runs:listChanged')
  })
}

// ── lifecycle ─────────────────────────────────────────────────────────────────

app.whenReady().then(() => {
  openDb()
  markStaleRunsInterrupted()
  scanExistingRuns()

  stopWatcher = watchRuns((runDir) => {
    pushUpdate(runDir)
    mainWindow?.webContents.send('runs:listChanged')
  })

  registerIpc()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  killActiveOnQuit()
  stopWatcher?.()
  closeDb()
})
