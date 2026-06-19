"use strict";
const electron = require("electron");
const path = require("path");
const Database = require("better-sqlite3");
const fs = require("fs");
const chokidar = require("chokidar");
const child_process = require("child_process");
let db;
function openDb() {
  if (db) return db;
  const dbPath = path.join(electron.app.getPath("userData"), "runs.sqlite");
  db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
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
  `);
  return db;
}
function closeDb() {
  db?.close();
}
function upsertRun(run_dir, run_name, status, pid, started_at, best_accuracy, node_count) {
  openDb().prepare(`
    INSERT INTO runs (run_dir, run_name, status, pid, started_at, best_accuracy, node_count)
    VALUES (@run_dir, @run_name, @status, @pid, @started_at, @best_accuracy, @node_count)
    ON CONFLICT(run_dir) DO UPDATE SET
      status = excluded.status,
      pid = excluded.pid,
      best_accuracy = excluded.best_accuracy,
      node_count = excluded.node_count
  `).run({ run_dir, run_name, status, pid, started_at, best_accuracy, node_count });
}
function setRunStatus(run_dir, status, finished_at = null) {
  openDb().prepare(`
    UPDATE runs SET status = ?, finished_at = ? WHERE run_dir = ?
  `).run(status, finished_at, run_dir);
}
function markStaleRunsInterrupted() {
  openDb().prepare(`
    UPDATE runs SET status = 'interrupted', finished_at = datetime('now')
    WHERE status = 'running'
  `).run();
}
function listRuns() {
  return openDb().prepare(`
    SELECT * FROM runs ORDER BY run_name ASC, started_at DESC
  `).all();
}
function getRun(run_dir) {
  return openDb().prepare(`SELECT * FROM runs WHERE run_dir = ?`).get(run_dir) ?? null;
}
function upsertNode(n) {
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
  `).run(n);
}
function getNodes(run_dir) {
  return openDb().prepare(`
    SELECT * FROM nodes WHERE run_dir = ? ORDER BY round ASC, timestamp ASC
  `).all(run_dir);
}
const RUNS_DIR = path.resolve(__dirname, "..", "..", "..", "..", "backend", "runs");
function getRunsDir() {
  return RUNS_DIR;
}
function getBackendDir() {
  return path.resolve(RUNS_DIR, "..");
}
function readMeta(runDir) {
  const metaPath = path.join(runDir, "meta.json");
  if (!fs.existsSync(metaPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(metaPath, "utf8"));
  } catch {
    return null;
  }
}
function ingestArchive(runDir) {
  const archivePath = path.join(runDir, "archive.json");
  if (!fs.existsSync(archivePath)) return;
  let nodes;
  try {
    nodes = JSON.parse(fs.readFileSync(archivePath, "utf8"));
  } catch {
    return;
  }
  const meta = readMeta(runDir);
  const run_name = meta?.run_name ?? path.basename(runDir);
  const started_at = meta?.started_at ?? "";
  const existing = getRun(runDir);
  const best_accuracy = nodes.reduce((m, n) => Math.max(m, n.accuracy ?? 0), 0);
  upsertRun(
    runDir,
    run_name,
    existing?.status ?? "done",
    existing?.pid ?? null,
    started_at,
    best_accuracy,
    nodes.length
  );
  for (const n of nodes) {
    const row = {
      run_dir: runDir,
      node_id: n.node_id,
      parent_id: n.parent_id ?? null,
      round: n.round ?? 0,
      accuracy: n.accuracy ?? 0,
      correct: n.metadata?.correct ?? n.correct ?? 0,
      total: n.metadata?.total ?? n.total ?? 0,
      compiles: n.compiles ? 1 : 0,
      model: n.model ?? "",
      timestamp: n.timestamp ?? "",
      reasoning: n.reasoning ?? "",
      novelty: n.novelty ?? ""
    };
    upsertNode(row);
  }
}
function scanExistingRuns() {
  if (!fs.existsSync(RUNS_DIR)) return;
  for (const entry of fs.readdirSync(RUNS_DIR, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    ingestArchive(path.join(RUNS_DIR, entry.name));
  }
}
function watchRuns(onUpdate) {
  const watcher = chokidar.watch(path.join(RUNS_DIR, "*", "archive.json"), {
    ignoreInitial: true,
    awaitWriteFinish: { stabilityThreshold: 300, pollInterval: 100 }
  });
  watcher.on("add", (filePath) => {
    const runDir = path.dirname(filePath);
    ingestArchive(runDir);
    onUpdate(runDir);
  });
  watcher.on("change", (filePath) => {
    const runDir = path.dirname(filePath);
    ingestArchive(runDir);
    onUpdate(runDir);
  });
  return () => watcher.close();
}
let activeChild = null;
let activeRunDir = null;
function isRunning() {
  return activeChild !== null;
}
function validateRunName(runName) {
  const backendDir = getBackendDir();
  const required = ["baselines", "evaluators", "improvement_agents"];
  for (const folder of required) {
    const dir = path.join(backendDir, folder, runName);
    if (!fs.existsSync(dir)) {
      return `Missing folder: backend/${folder}/${runName}`;
    }
  }
  return null;
}
function startRun(runName, onUpdate) {
  if (activeChild) {
    return { ok: false, error: "A run is already active." };
  }
  const validationError = validateRunName(runName);
  if (validationError) {
    return { ok: false, error: validationError };
  }
  const ts = (/* @__PURE__ */ new Date()).toISOString().replace(/[:.]/g, "-").slice(0, 19).replace("T", "_");
  const runDir = path.join(getRunsDir(), ts);
  fs.mkdirSync(path.join(runDir, "nodes"), { recursive: true });
  const startedAt = (/* @__PURE__ */ new Date()).toISOString();
  fs.writeFileSync(
    path.join(runDir, "meta.json"),
    JSON.stringify({ run_name: runName, started_at: startedAt }, null, 2)
  );
  upsertRun(runDir, runName, "running", null, startedAt, 0, 0);
  const child = child_process.spawn(
    "python3",
    ["run_self_improvement.py", "--run-name", runName, "--run-dir", runDir],
    {
      cwd: getBackendDir(),
      detached: true,
      stdio: "ignore"
    }
  );
  child.unref();
  upsertRun(runDir, runName, "running", child.pid ?? null, startedAt, 0, 0);
  activeChild = child;
  activeRunDir = runDir;
  child.on("exit", (code) => {
    const status = code === 0 ? "done" : "failed";
    setRunStatus(runDir, status, (/* @__PURE__ */ new Date()).toISOString());
    activeChild = null;
    activeRunDir = null;
    onUpdate(runDir);
  });
  return { ok: true, run_dir: runDir };
}
function stopRun() {
  if (!activeChild || activeChild.pid == null) return;
  try {
    process.kill(-activeChild.pid, "SIGTERM");
  } catch {
    try {
      activeChild.kill("SIGTERM");
    } catch {
    }
  }
  if (activeRunDir) {
    setRunStatus(activeRunDir, "interrupted", (/* @__PURE__ */ new Date()).toISOString());
  }
  activeChild = null;
  activeRunDir = null;
}
function killActiveOnQuit() {
  if (!activeChild || activeChild.pid == null) return;
  try {
    process.kill(-activeChild.pid, "SIGKILL");
  } catch {
    try {
      activeChild.kill("SIGKILL");
    } catch {
    }
  }
  if (activeRunDir) {
    setRunStatus(activeRunDir, "interrupted", (/* @__PURE__ */ new Date()).toISOString());
  }
}
let mainWindow = null;
let stopWatcher = null;
function createWindow() {
  mainWindow = new electron.BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    titleBarStyle: "hiddenInset",
    webPreferences: {
      preload: path.join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });
  if (process.env["ELECTRON_RENDERER_URL"]) {
    mainWindow.loadURL(process.env["ELECTRON_RENDERER_URL"]);
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));
  }
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    electron.shell.openExternal(url);
    return { action: "deny" };
  });
}
function pushUpdate(runDir) {
  const run = getRun(runDir);
  mainWindow?.webContents.send("run:updated", run);
}
function registerIpc() {
  electron.ipcMain.handle("runs:list", () => listRuns());
  electron.ipcMain.handle("runs:get", (_e, runDir) => getRun(runDir));
  electron.ipcMain.handle("nodes:get", (_e, runDir) => getNodes(runDir));
  electron.ipcMain.handle("runs:isRunning", () => isRunning());
  electron.ipcMain.handle("runs:start", (_e, runName) => {
    return startRun(runName, (runDir) => {
      pushUpdate(runDir);
      mainWindow?.webContents.send("runs:listChanged");
    });
  });
  electron.ipcMain.handle("runs:stop", () => {
    stopRun();
    mainWindow?.webContents.send("runs:listChanged");
  });
}
electron.app.whenReady().then(() => {
  openDb();
  markStaleRunsInterrupted();
  scanExistingRuns();
  stopWatcher = watchRuns((runDir) => {
    pushUpdate(runDir);
    mainWindow?.webContents.send("runs:listChanged");
  });
  registerIpc();
  createWindow();
  electron.app.on("activate", () => {
    if (electron.BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});
electron.app.on("window-all-closed", () => {
  if (process.platform !== "darwin") electron.app.quit();
});
electron.app.on("before-quit", () => {
  killActiveOnQuit();
  stopWatcher?.();
  closeDb();
});
