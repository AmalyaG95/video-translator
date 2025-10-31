const { contextBridge, ipcRenderer } = require("electron");

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld("electronAPI", {
  selectVideoFile: () => ipcRenderer.invoke("select-video-file"),
  getAppVersion: () => ipcRenderer.invoke("get-app-version"),
  showErrorDialog: (title, content) =>
    ipcRenderer.invoke("show-error-dialog", title, content),

  // Backend API calls - Updated for NestJS API (port 3001)
  uploadVideo: (filePath) =>
    fetch("http://localhost:3001/api/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_path: filePath }),
    }).then((res) => res.json()),

  startTranslation: (config) =>
    fetch("http://localhost:3001/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }).then((res) => res.json()),

  getProgress: (jobId) =>
    fetch(`http://localhost:3001/api/progress/${jobId}`).then((res) =>
      res.json()
    ),

  downloadResult: (jobId) =>
    fetch(`http://localhost:3001/api/download/${jobId}`).then((res) =>
      res.blob()
    ),

  getPreview: (jobId) =>
    fetch(`http://localhost:3001/api/preview/${jobId}`).then((res) =>
      res.blob()
    ),
});
