"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("api", {
  listRuns: () => electron.ipcRenderer.invoke("runs:list"),
  getRun: (runDir) => electron.ipcRenderer.invoke("runs:get", runDir),
  getNodes: (runDir) => electron.ipcRenderer.invoke("nodes:get", runDir),
  isRunning: () => electron.ipcRenderer.invoke("runs:isRunning"),
  startRun: (runName) => electron.ipcRenderer.invoke("runs:start", runName),
  stopRun: () => electron.ipcRenderer.invoke("runs:stop"),
  onRunUpdated: (cb) => {
    electron.ipcRenderer.on("run:updated", (_e, run) => cb(run));
  },
  onListChanged: (cb) => {
    electron.ipcRenderer.on("runs:listChanged", () => cb());
  },
  removeAllListeners: () => {
    electron.ipcRenderer.removeAllListeners("run:updated");
    electron.ipcRenderer.removeAllListeners("runs:listChanged");
  }
});
