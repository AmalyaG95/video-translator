const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

// Import standalone launcher for bundled AppImage
let standaloneLauncher = null;
try {
  standaloneLauncher = require("./standalone-launcher");
} catch (e) {
  // Launcher not available (development mode)
  console.log("[Electron] Standalone launcher not available, using development mode");
}

let mainWindow;
let backendProcess;

// Security settings
const isPackaged = app.isPackaged || process.env.APPIMAGE !== undefined;
// Only consider it dev mode if explicitly set AND not packaged
const isDev = !isPackaged && process.env.NODE_ENV === "development";

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

  // Don't load URL here - it will be loaded after services start in standalone mode
  // For dev mode, load immediately
  if (isDev) {
    mainWindow.loadURL("http://localhost:3000");
  }
  // For standalone/packaged mode, URL will be loaded after services start

  // Show window when ready to prevent flickering
  // But in standalone mode, we'll show it immediately with loading screen
  if (!isPackaged || isDev) {
    mainWindow.once("ready-to-show", () => {
      mainWindow.show();
    });
  }

  // Handle page load errors
  mainWindow.webContents.on("did-fail-load", (event, errorCode, errorDescription, validatedURL) => {
    console.error("[Electron] Failed to load:", validatedURL, errorCode, errorDescription);
    // Show error page if frontend fails to load
    if (!isDev) {
      mainWindow.loadURL(`data:text/html;charset=utf-8,
        <html>
          <head><title>Video Translator - Error</title></head>
          <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
            <h1>Failed to Load Application</h1>
            <p>Error: ${errorDescription || errorCode}</p>
            <p>URL: ${validatedURL}</p>
            <p>Please check the console for more details.</p>
            <p>Make sure all backend services are running.</p>
          </body>
        </html>
      `);
    }
  });

  // Log console messages from renderer
  mainWindow.webContents.on("console-message", (event, level, message, line, sourceId) => {
    console.log(`[Renderer ${level}]:`, message);
  });

  // Never open devtools in packaged/production builds
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
app.commandLine.appendSwitch("--allow-running-insecure-content");

/**
 * Detect system capabilities and configure Electron accordingly
 * Uses device capabilities efficiently - only disables what's not available
 */

// Detect containerized environment
const isDocker = process.env.DOCKER_CONTAINER === "true" || fs.existsSync("/.dockerenv");

// Detect display server capabilities
const hasX11 = process.env.DISPLAY && process.env.DISPLAY.startsWith(":");
const hasWayland = !!process.env.WAYLAND_DISPLAY || !!process.env.XDG_RUNTIME_DIR;
const hasDisplay = hasX11 || hasWayland;

// Check for GPU device files (Linux)
let hasGPUDriver = false;
if (process.platform === "linux") {
  try {
    // Check for DRM devices (indicates GPU hardware)
    const drmDevices = ["/dev/dri/card0", "/dev/dri/card1", "/dev/dri/renderD128", "/dev/dri/renderD129"];
    hasGPUDriver = drmDevices.some(device => fs.existsSync(device));
  } catch (e) {
    // If we can't check, assume no GPU driver
    hasGPUDriver = false;
  }
}

// Platform-specific GPU detection
const platformHasGPU = 
  process.platform === "darwin" ||  // macOS has GPU
  process.platform === "win32" ||   // Windows has GPU
  (process.platform === "linux" && hasGPUDriver); // Linux with GPU driver

// Determine if we should use GPU
const shouldUseGPU = !isDocker && platformHasGPU && hasDisplay;

// Determine if we should use Vulkan (only if GPU available and not in Docker)
const shouldUseVulkan = shouldUseGPU && !isDocker;

// Log detected capabilities
console.log("[Electron] System capabilities:", {
  platform: process.platform,
  isDocker,
  hasDisplay,
  hasX11,
  hasWayland,
  hasGPUDriver,
  platformHasGPU,
  shouldUseGPU,
  shouldUseVulkan
});

// Apply configuration based on capabilities
// IMPORTANT: These flags must be set BEFORE app.whenReady()
if (isDocker) {
  // Docker environment - use software rendering
  console.log("[Electron] Docker detected - using software rendering");
  
  // Required sandbox flags for Docker
  app.commandLine.appendSwitch("--no-sandbox");
  app.commandLine.appendSwitch("--disable-setuid-sandbox");
  app.commandLine.appendSwitch("--disable-dev-shm-usage");
  
  // Completely disable GPU process FIRST (before other GPU flags)
  app.commandLine.appendSwitch("--disable-gpu-process");
  app.commandLine.appendSwitch("--in-process-gpu");
  
  // Software rendering only (no GPU in Docker)
  app.commandLine.appendSwitch("--disable-gpu");
  app.commandLine.appendSwitch("--use-gl", "swiftshader");
  app.commandLine.appendSwitch("--use-angle", "swiftshader");
  
  // Disable ANGLE/Vulkan completely - must be done early
  app.commandLine.appendSwitch("--disable-features", "Vulkan,VulkanFromANGLE,UseChromeOSDirectVideoDecoder,DefaultANGLEVulkan,UseSkiaRenderer");
  app.commandLine.appendSwitch("--disable-accelerated-2d-canvas");
  app.commandLine.appendSwitch("--disable-accelerated-video-decode");
  app.commandLine.appendSwitch("--disable-accelerated-video-encode");
  app.commandLine.appendSwitch("--disable-gpu-compositing");
  
  // Force software rasterization
  app.commandLine.appendSwitch("--enable-software-rasterizer");
  
} else if (shouldUseGPU) {
  // Native system with GPU - use hardware acceleration
  console.log("[Electron] GPU detected - using hardware acceleration");
  
  // Enable GPU features
  // Don't disable GPU - let Electron use it
  // Don't disable acceleration - use it for better performance
  
  // Only disable Vulkan if not supported
  if (!shouldUseVulkan) {
    app.commandLine.appendSwitch("--disable-features", "Vulkan,VulkanFromANGLE");
    console.log("[Electron] Vulkan not available - using OpenGL");
  } else {
    console.log("[Electron] Using Vulkan for better performance");
  }
  
} else {
  // Native system without GPU - use software rendering
  console.log("[Electron] No GPU detected - using software rendering");
  
  app.commandLine.appendSwitch("--disable-gpu");
  app.commandLine.appendSwitch("--use-gl", "swiftshader");
  app.commandLine.appendSwitch("--disable-features", "Vulkan,VulkanFromANGLE");
  app.commandLine.appendSwitch("--disable-accelerated-2d-canvas");
  app.commandLine.appendSwitch("--disable-accelerated-video-decode");
}

app.whenReady().then(async () => {
  // Check if running as standalone AppImage
  const isStandalone = process.env.APPIMAGE !== undefined || (isPackaged && !process.env.SKIP_BACKEND && !isDev);
  
  if (isStandalone && standaloneLauncher) {
    // Standalone mode - start all bundled services BEFORE creating window
    console.log("[Electron] Starting in standalone mode...");
    try {
      // Show a loading message immediately
      createWindow();
      
      // Load loading screen from file (more reliable than data: URL)
      const loadingPath = path.join(__dirname, "loading.html");
      if (fs.existsSync(loadingPath)) {
        mainWindow.loadFile(loadingPath);
      } else {
        // Fallback to data URL if file doesn't exist
        const loadingHtml = `data:text/html;charset=utf-8,${encodeURIComponent(`
          <!DOCTYPE html>
          <html>
          <head>
            <meta charset="utf-8">
            <title>Video Translator - Starting...</title>
            <style>
              body { font-family: Arial, sans-serif; padding: 60px 40px; text-align: center; background: #f8fafc; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }
              h1 { color: #1a202c; font-size: 32px; margin-bottom: 30px; }
              .spinner { border: 4px solid #e2e8f0; border-top: 4px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 20px auto 30px; }
              @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
              p { color: #4a5568; font-size: 16px; margin: 10px 0; }
              .subtitle { color: #718096; font-size: 14px; margin-top: 30px; }
            </style>
          </head>
          <body>
            <h1>Video Translator</h1>
            <div class="spinner"></div>
            <p>Starting services...</p>
            <p>Please wait while we initialize the application.</p>
            <p class="subtitle">This may take 10-20 seconds on first launch</p>
          </body>
          </html>
        `)}`;
        mainWindow.loadURL(loadingHtml);
      }
      mainWindow.show();
      
      // Start services with timeout (longer for first launch - Python may need to download models)
      const serviceStartTimeout = 180000; // 3 minutes max (Python models can take time on first launch)
      const startServicesPromise = standaloneLauncher.startAllServices();
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error("Services took too long to start (3 minute timeout). This may be normal on first launch if Python models need to download. Check console for details.")), serviceStartTimeout);
      });
      
      await Promise.race([startServicesPromise, timeoutPromise]);
      console.log("[Electron] All services started successfully");
      
      // Services are ready, load the frontend
      const startUrl = "http://localhost:3000";
      console.log("[Electron] Loading frontend at:", startUrl);
      
      // Double-check frontend is ready before loading
      let frontendReady = false;
      for (let i = 0; i < 10; i++) {
        try {
          const http = require("http");
          await new Promise((resolve, reject) => {
            const req = http.get(startUrl, { timeout: 2000 }, (res) => {
              resolve(true);
            });
            req.on("error", reject);
            req.on("timeout", () => {
              req.destroy();
              reject(new Error("Timeout"));
            });
          });
          frontendReady = true;
          break;
        } catch (e) {
          if (i < 9) {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
      }
      
      if (frontendReady) {
        mainWindow.loadURL(startUrl);
      } else {
        throw new Error("Frontend server did not respond after services started");
      }
    } catch (error) {
      console.error("[Electron] Failed to start bundled services:", error);
      if (!mainWindow) {
        createWindow();
      }
      // Format error message for display (replace newlines with <br>)
      const formattedError = (error.message || "Unknown error")
        .replace(/\n/g, '<br>')
        .replace(/  /g, '&nbsp;&nbsp;');
      
      mainWindow.loadURL(`data:text/html;charset=utf-8,
        <html>
          <head>
            <title>Video Translator - Error</title>
            <style>
              body { font-family: Arial, sans-serif; padding: 40px; text-align: center; background: #f8fafc; }
              .error-box { background: white; border: 2px solid #e53e3e; border-radius: 8px; padding: 30px; max-width: 800px; margin: 0 auto; text-align: left; }
              h1 { color: #c53030; margin-top: 0; }
              .error-details { background: #fff5f5; padding: 15px; border-radius: 4px; margin: 20px 0; font-family: monospace; font-size: 12px; white-space: pre-wrap; }
              .fix-steps { background: #ebf8ff; padding: 15px; border-radius: 4px; margin: 20px 0; border-left: 4px solid #3182ce; }
              .fix-steps h3 { margin-top: 0; color: #2c5282; }
              .fix-steps code { background: #edf2f7; padding: 2px 6px; border-radius: 3px; }
            </style>
          </head>
          <body>
            <div class="error-box">
              <h1>Failed to Start Services</h1>
              <div class="error-details">${formattedError}</div>
              <div class="fix-steps">
                <h3>How to Fix:</h3>
                <p>1. Check the console output for detailed error messages</p>
                <p>2. Make sure Python 3.9+ is installed: <code>python3 --version</code></p>
                <p>3. If the error mentions missing dependencies, rebuild the venv:</p>
                <p style="margin-left: 20px;">
                  <code>rm -rf backend-python-ml-v2/venv</code><br>
                  <code>./prepare-python-venv.sh</code><br>
                  <code>./build-appimage.sh</code>
                </p>
              </div>
            </div>
          </body>
        </html>
      `);
      mainWindow.show();
      dialog.showErrorBox(
        "Failed to Start Services",
        `Could not start backend services: ${error.message}\n\nPlease check the console for more details.`
      );
    }
  } else {
    // Development mode or non-standalone - create window immediately
    createWindow();
    if (!process.env.SKIP_BACKEND && !isDev) {
      startBackend();
    }
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  stopBackend();
  if (standaloneLauncher) {
    standaloneLauncher.stopAllServices();
  }
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  stopBackend();
  if (standaloneLauncher) {
    standaloneLauncher.stopAllServices();
  }
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
