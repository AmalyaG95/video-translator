const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

let mainWindow;
let backendProcess;

// Security settings
const isDev = process.env.NODE_ENV === "development";

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      webSecurity: false, // Disable web security for development
      allowRunningInsecureContent: true, // Allow insecure content
      experimentalFeatures: true, // Enable experimental features
      preload: path.join(__dirname, "preload.js"),
    },
    icon: path.join(__dirname, "assets", "logo.png"),
    title: "Video Translator",
    backgroundColor: "#f8fafc",
    show: false, // Don't show until ready
  });

  // Load the Next.js app
  const startUrl = isDev
    ? "http://localhost:3000"
    : `file://${path.join(__dirname, "../frontend/out/index.html")}`;

  mainWindow.loadURL(startUrl);

  // Show window when ready to prevent flickering
  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
  });

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    // Open external links in default browser
    if (url.startsWith("http://") || url.startsWith("https://")) {
      require("electron").shell.openExternal(url);
      return { action: "deny" };
    }
    return { action: "allow" };
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function startBackend() {
  const backendPath = path.join(__dirname, "../backend");
  const pythonPath = process.platform === "win32" ? "python" : "python3";

  backendProcess = spawn(
    pythonPath,
    ["-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
    {
      cwd: backendPath,
      stdio: "inherit",
    }
  );

  backendProcess.on("error", (err) => {
    console.error("Failed to start backend:", err);
  });
}

function stopBackend() {
  if (backendProcess) {
    backendProcess.kill();
  }
}

// Disable CORS and security for development
app.commandLine.appendSwitch("--disable-web-security");
app.commandLine.appendSwitch("--disable-features", "VizDisplayCompositor");
app.commandLine.appendSwitch("--allow-running-insecure-content");
app.commandLine.appendSwitch("--disable-site-isolation-trials");

app.whenReady().then(() => {
  createWindow();

  // Only start backend if not using Docker (when SKIP_BACKEND env var is not set)
  if (!process.env.SKIP_BACKEND && !isDev) {
    startBackend();
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  stopBackend();
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  stopBackend();
});

// IPC handlers
ipcMain.handle("select-video-file", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile"],
    filters: [
      { name: "Video Files", extensions: ["mp4", "avi", "mov", "mkv", "webm"] },
    ],
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle("get-app-version", () => {
  return app.getVersion();
});

ipcMain.handle("show-error-dialog", async (event, title, content) => {
  await dialog.showErrorBox(title, content);
});
