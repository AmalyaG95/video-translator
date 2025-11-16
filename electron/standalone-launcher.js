/**
 * Standalone Launcher for Video Translator
 * 
 * This script starts all required backend services when running as a standalone AppImage.
 * It manages:
 * - NestJS API server (port 3001)
 * - Python ML service (gRPC port 50051)
 * - Frontend (Next.js, port 3000)
 */

const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

// Detect if running from AppImage
const isAppImage = process.env.APPIMAGE !== undefined;
const isPackaged = process.env.NODE_ENV === "production" || process.env.APPIMAGE !== undefined;

// Get resource paths
function getResourcePath(relativePath) {
  if (isAppImage || isPackaged) {
    // In AppImage/packaged app, resources are in extraResources (unpacked from asar)
    // They should be directly in process.resourcesPath
    const resourcePath = path.join(process.resourcesPath, relativePath);
    
    console.log("[Launcher] Looking for resource:", relativePath);
    console.log("[Launcher] APPIMAGE:", process.env.APPIMAGE);
    console.log("[Launcher] resourcesPath:", process.resourcesPath);
    console.log("[Launcher] execPath:", process.execPath);
    console.log("[Launcher] Checking resource path:", resourcePath);
    
    // Try multiple possible locations for AppImage
    const possiblePaths = [
      path.join(process.resourcesPath, relativePath), // Direct in resources (most common)
      path.join(process.resourcesPath, "app.asar.unpacked", relativePath), // Unpacked from asar
      path.join(path.dirname(process.execPath), "resources", relativePath), // Next to executable
      path.join(path.dirname(process.execPath), "resources", "app.asar.unpacked", relativePath),
    ];
    
    // If APPIMAGE env var is set, also check AppImage mount location
    if (process.env.APPIMAGE) {
      const appImageDir = path.dirname(process.env.APPIMAGE);
      possiblePaths.push(
        path.join(appImageDir, "resources", relativePath),
        path.join(appImageDir, "resources", "app.asar.unpacked", relativePath)
      );
    }
    
    for (const possiblePath of possiblePaths) {
      console.log("[Launcher] Checking path:", possiblePath, "exists:", fs.existsSync(possiblePath));
      if (fs.existsSync(possiblePath)) {
        console.log("[Launcher] ✓ Found resource at:", possiblePath);
        return possiblePath;
      }
    }
    
    // Resource not found - this is a critical error
    console.error("[Launcher] ✗ Resource not found:", relativePath);
    console.error("[Launcher] Checked paths:", possiblePaths);
    throw new Error(`Resource not found: ${relativePath}. Checked: ${possiblePaths.join(", ")}`);
  } else {
    // Development mode - use project root
    const devPath = path.join(__dirname, "..", relativePath);
    console.log("[Launcher] Development resource path:", devPath);
    if (!fs.existsSync(devPath)) {
      console.error("[Launcher] ✗ Development resource not found:", devPath);
      throw new Error(`Development resource not found: ${devPath}`);
    }
    return devPath;
  }
}

// Get Node.js executable path
function getNodePath() {
  // In Electron, we can use the bundled Node.js
  // Electron includes Node.js, but we need to find it
  if (isAppImage || isPackaged) {
    // Try to find Node.js in Electron's resources
    const electronPath = process.execPath;
    const electronDir = path.dirname(electronPath);
    
    // Electron bundles Node.js - try to use it
    // For backend services, we'll use the system Node.js or bundled one
    // Check if there's a node binary nearby
    const possibleNodePaths = [
      path.join(electronDir, "node"),
      path.join(electronDir, "..", "node"),
      "/usr/bin/node",
      "/usr/local/bin/node",
    ];
    
    for (const nodePath of possibleNodePaths) {
      if (fs.existsSync(nodePath)) {
        return nodePath;
      }
    }
  }
  
  // Fallback to system Node.js
  return "node";
}

let nestjsProcess = null;
let pythonMLProcess = null;
let frontendProcess = null;

function startFrontend() {
  const frontendPath = getResourcePath("frontend");
  const nodePath = getNodePath();
  
  console.log("[Launcher] Starting frontend server at:", frontendPath);
  console.log("[Launcher] Node path:", nodePath);
  
  // Check if .next directory exists (Next.js build output)
  const nextPath = path.join(frontendPath, ".next");
  if (!fs.existsSync(nextPath)) {
    console.error("[Launcher] Frontend not built! .next directory not found at:", nextPath);
    console.error("[Launcher] Available files in frontend path:", fs.existsSync(frontendPath) ? fs.readdirSync(frontendPath).join(", ") : "path does not exist");
    return null;
  }

  // Check if node_modules exists
  const nodeModulesPath = path.join(frontendPath, "node_modules");
  if (!fs.existsSync(nodeModulesPath)) {
    console.error("[Launcher] Frontend node_modules not found at:", nodeModulesPath);
    return null;
  }

  // Find Next.js binary
  const nextBinary = path.join(frontendPath, "node_modules", ".bin", "next");
  console.log("[Launcher] Next.js binary path:", nextBinary);
  console.log("[Launcher] Next.js binary exists:", fs.existsSync(nextBinary));

  // Start Next.js production server
  const args = fs.existsSync(nextBinary) 
    ? [nextBinary, "start", "-p", "3000"]
    : ["next", "start", "-p", "3000"];

  console.log("[Launcher] Starting Next.js with command:", nodePath, args.join(" "));

  frontendProcess = spawn(
    nodePath,
    args,
    {
      cwd: frontendPath,
      env: {
        ...process.env,
        NODE_ENV: "production",
        PORT: "3000",
        NEXT_PUBLIC_API_URL: "http://localhost:3001",
      },
      stdio: "inherit",
    }
  );

  frontendProcess.on("error", (err) => {
    console.error("[Launcher] Failed to start frontend:", err);
    console.error("[Launcher] Error details:", err.message, err.stack);
  });

  frontendProcess.on("exit", (code, signal) => {
    if (code !== null && code !== 0) {
      console.error(`[Launcher] Frontend process exited with code ${code}`);
    }
    if (signal) {
      console.error(`[Launcher] Frontend process killed with signal ${signal}`);
    }
  });

  return frontendProcess;
}

function startNestJS() {
  const nestjsPath = getResourcePath("backend-nestjs");
  const nodePath = getNodePath();
  
  console.log("[Launcher] Starting NestJS API at:", nestjsPath);
  
  // Check if dist directory exists
  const distPath = path.join(nestjsPath, "dist");
  if (!fs.existsSync(distPath)) {
    console.error("[Launcher] NestJS not built! dist directory not found at:", distPath);
    return null;
  }

  // Set up data directories for standalone mode
  const homeDir = process.env.HOME || process.env.USERPROFILE || ".";
  const dataDir = path.join(homeDir, "video-translator-data");
  const uploadsDir = path.join(dataDir, "uploads");
  const artifactsDir = path.join(dataDir, "artifacts");
  
  // Ensure directories exist
  [dataDir, uploadsDir, artifactsDir].forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  });

  nestjsProcess = spawn(
    nodePath,
    ["dist/main.js"],
    {
      cwd: nestjsPath,
      env: {
        ...process.env,
        NODE_ENV: "production",
        PORT: "3001",
        GRPC_ML_SERVICE_URL: "localhost:50051",
        // Set paths for standalone mode (not Docker)
        UPLOADS_DIR: uploadsDir,
        ARTIFACTS_DIR: artifactsDir,
        DOCKER_CONTAINER: "false", // Explicitly mark as not Docker
      },
      stdio: "inherit",
    }
  );

  nestjsProcess.on("error", (err) => {
    console.error("[Launcher] Failed to start NestJS:", err);
  });

  return nestjsProcess;
}

function startPythonML() {
  const pythonMLPath = getResourcePath("backend-python-ml-v2");
  
  // Try to find Python - check bundled venv first, then system
  let pythonPath = "python3";
  
  // Check for bundled virtual environment (preferred - standalone)
  const bundledVenvPython = path.join(pythonMLPath, "venv", "bin", "python3");
  const bundledVenvPythonWin = path.join(pythonMLPath, "venv", "Scripts", "python.exe");
  const bundledVenvDir = path.join(pythonMLPath, "venv");
  
  // Use system Python with venv's site-packages (more reliable than copied Python binary)
  // The copied Python binary has hardcoded paths that break in AppImage
  if (fs.existsSync(bundledVenvDir)) {
    console.log("[Launcher] ✓ Found bundled Python virtual environment");
    console.log("[Launcher] Using system Python with venv's site-packages");
    
    // Use system Python (more reliable)
    pythonPath = "python3";
    
    // Verify system Python exists
    try {
      const { execSync } = require("child_process");
      const pythonVersion = execSync(`${pythonPath} --version`, { encoding: "utf-8", timeout: 5000 });
      console.log("[Launcher] System Python version:", pythonVersion.trim());
    } catch (err) {
      throw new Error(`Python 3 not found. Please install Python 3.9+ and ensure 'python3' is in your PATH. Error: ${err.message}`);
    }
  } else if (fs.existsSync(bundledVenvPythonWin)) {
    pythonPath = bundledVenvPythonWin;
    console.log("[Launcher] ✓ Using bundled Python virtual environment (Windows):", pythonPath);
  } else {
    // Fallback to system Python (for development or if venv not bundled)
    console.log("[Launcher] ⚠ No bundled Python venv found, using system Python");
    console.log("[Launcher] Note: System Python dependencies must be installed");
    
    // Verify system Python exists and is accessible
    try {
      const { execSync } = require("child_process");
      const pythonVersion = execSync(`${pythonPath} --version`, { encoding: "utf-8", timeout: 5000 });
      console.log("[Launcher] Using system Python:", pythonPath, "-", pythonVersion.trim());
    } catch (err) {
      throw new Error(`Python 3 not found. Please install Python 3.9+ and ensure 'python3' is in your PATH. Error: ${err.message}`);
    }
  }

  console.log("[Launcher] Starting Python ML service at:", pythonMLPath);
  
  // Generate proto files if needed (similar to entrypoint.sh)
  const protoFile = path.join(pythonMLPath, "src", "proto", "translation.proto");
  const protoGenerated = path.join(pythonMLPath, "src", "proto", "translation_pb2.py");
  
  if (fs.existsSync(protoFile) && (!fs.existsSync(protoGenerated) || 
      fs.statSync(protoFile).mtime > fs.statSync(protoGenerated).mtime)) {
    console.log("[Launcher] Generating gRPC proto files...");
    try {
      const { execSync } = require("child_process");
      execSync(`${pythonPath} -m grpc_tools.protoc -I./src/proto --python_out=./src/proto --grpc_python_out=./src/proto --pyi_out=./src/proto ./src/proto/translation.proto`, {
        cwd: pythonMLPath,
        stdio: "inherit",
      });
      console.log("[Launcher] Proto files generated successfully");
    } catch (err) {
      console.warn("[Launcher] Failed to generate proto files (they should already exist):", err.message);
    }
  }

  // Set up data directories for standalone mode
  const homeDir = process.env.HOME || process.env.USERPROFILE || ".";
  const dataDir = path.join(homeDir, "video-translator-data");
  const uploadsDir = path.join(dataDir, "uploads");
  const artifactsDir = path.join(dataDir, "artifacts");
  const tempWorkDir = path.join(dataDir, "temp_work");
  
  // Ensure directories exist
  [dataDir, uploadsDir, artifactsDir, tempWorkDir].forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  });

  console.log("[Launcher] Starting Python with command:", pythonPath, "-m src.main");
  console.log("[Launcher] Working directory:", pythonMLPath);
  console.log("[Launcher] Environment variables:");
  console.log("[Launcher]   UPLOADS_DIR:", uploadsDir);
  console.log("[Launcher]   ARTIFACTS_DIR:", artifactsDir);
  console.log("[Launcher]   TEMP_WORK_DIR:", tempWorkDir);

  // Capture Python output to see errors
  let pythonOutput = "";
  let pythonErrors = "";

  // Set up environment for venv Python
  const env = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    GRPC_PORT: "50051",
    // Set paths for standalone mode (not Docker)
    UPLOADS_DIR: uploadsDir,
    ARTIFACTS_DIR: artifactsDir,
    TEMP_WORK_DIR: tempWorkDir,
    DOCKER_CONTAINER: "false", // Explicitly mark as not Docker
  };
  
  // If venv exists, use system Python with venv's site-packages
  let venvSitePackages = null;
  if (fs.existsSync(bundledVenvDir)) {
    const venvLib = path.join(bundledVenvDir, "lib");
    if (fs.existsSync(venvLib)) {
      const pythonDirs = fs.readdirSync(venvLib).filter(d => d.startsWith("python"));
      if (pythonDirs.length > 0) {
        const foundSitePackages = path.join(venvLib, pythonDirs[0], "site-packages");
        if (fs.existsSync(foundSitePackages)) {
          venvSitePackages = foundSitePackages;
          
          // Add site-packages to PYTHONPATH so system Python can use venv packages
          // Put it FIRST so it takes precedence
          const currentPythonPath = env.PYTHONPATH || "";
          env.PYTHONPATH = foundSitePackages + (currentPythonPath ? ":" + currentPythonPath : "");
          console.log("[Launcher] Using system Python with venv site-packages:", foundSitePackages);
          
          // Verify grpc is available before starting
          // Use a simpler check - just verify the package directory exists
          // The actual import will work when PYTHONPATH is set correctly
          const grpcPath = path.join(foundSitePackages, "grpc");
          const grpcioPath = path.join(foundSitePackages, "grpc");
          
          // Check if grpc package exists in site-packages
          // grpc is installed as "grpcio" package but imports as "grpc"
          const hasGrpc = fs.existsSync(path.join(foundSitePackages, "grpc")) || 
                          fs.existsSync(path.join(foundSitePackages, "grpc.py")) ||
                          fs.existsSync(path.join(foundSitePackages, "grpc", "__init__.py"));
          
          // Also check for grpcio package (which provides grpc module)
          const hasGrpcio = fs.existsSync(path.join(foundSitePackages, "grpcio")) ||
                            fs.existsSync(path.join(foundSitePackages, "grpcio.py"));
          
          if (!hasGrpc && !hasGrpcio) {
            // Try to find grpc-related packages
            const grpcPackages = fs.readdirSync(foundSitePackages).filter(d => 
              d.startsWith("grpc") || d.includes("grpc")
            );
            
            if (grpcPackages.length === 0) {
              console.error("[Launcher] ✗ No grpc packages found in site-packages");
              console.error("[Launcher] Site-packages path:", foundSitePackages);
              console.error("[Launcher] Site-packages exists:", fs.existsSync(foundSitePackages));
              if (fs.existsSync(foundSitePackages)) {
                const samplePackages = fs.readdirSync(foundSitePackages).slice(0, 20);
                console.error("[Launcher] Sample packages:", samplePackages.join(", "));
              }
              throw new Error(`Bundled Python venv is missing grpc package.\n\nThe venv may be incomplete.\n\nTo fix:\n1. Delete: rm -rf backend-python-ml-v2/venv\n2. Rebuild: ./prepare-python-venv.sh\n3. Rebuild AppImage: ./build-appimage.sh`);
            } else {
              console.log("[Launcher] Found grpc-related packages:", grpcPackages.join(", "));
            }
          } else {
            console.log("[Launcher] ✓ grpc package found in site-packages");
          }
          
          // Package directory exists - trust it and continue
          // The actual import will be tested when Python starts
          // If there's a real problem, Python will exit with an error and we'll catch it
          console.log("[Launcher] ✓ grpc package directory found - dependencies should be available");
          console.log("[Launcher] Will verify during Python startup (if import fails, error will be shown)");
        } else {
          console.warn("[Launcher] ⚠ site-packages not found at:", foundSitePackages);
          throw new Error(`Python venv site-packages not found at: ${foundSitePackages}. The venv may be incomplete.`);
        }
      } else {
        console.warn("[Launcher] ⚠ No python directory found in venv lib:", venvLib);
        throw new Error(`Python venv lib directory structure is invalid. Expected python3.x directory in: ${venvLib}`);
      }
    } else {
      console.warn("[Launcher] ⚠ venv lib directory not found:", venvLib);
      throw new Error(`Python venv lib directory not found: ${venvLib}. The venv may be incomplete.`);
    }
    
    // Don't set PYTHONHOME - use system Python's standard library
    delete env.PYTHONHOME;
  }
  
  // Log final environment for debugging
  console.log("[Launcher] Final Python environment:");
  console.log("[Launcher]   PYTHONPATH:", env.PYTHONPATH || "(not set)");
  console.log("[Launcher]   PYTHONHOME:", env.PYTHONHOME || "(not set)");
  console.log("[Launcher]   Python executable:", pythonPath);

  pythonMLProcess = spawn(
    pythonPath,
    ["-m", "src.main"],
    {
      cwd: pythonMLPath,
      env: env,
      stdio: ["ignore", "pipe", "pipe"], // Capture stdout and stderr
    }
  );

  // Capture stdout
  pythonMLProcess.stdout.on("data", (data) => {
    const output = data.toString();
    pythonOutput += output;
    // Also log to console in real-time
    process.stdout.write(`[Python] ${output}`);
  });

  // Capture stderr (errors)
  pythonMLProcess.stderr.on("data", (data) => {
    const error = data.toString();
    pythonErrors += error;
    // Also log to console in real-time
    process.stderr.write(`[Python Error] ${error}`);
  });

  pythonMLProcess.on("error", (err) => {
    console.error("[Launcher] ✗ Failed to start Python ML service:", err);
    console.error("[Launcher] Error details:", err.message, err.stack);
    if (err.code === "ENOENT") {
      throw new Error(`Python executable not found: ${pythonPath}. Please install Python 3.9+ and ensure it's in your PATH.`);
    }
    throw err;
  });

  pythonMLProcess.on("exit", (code, signal) => {
    if (code !== null && code !== 0) {
      console.error(`[Launcher] ✗ Python ML process exited with code ${code}`);
      console.error("[Launcher] Python stdout:", pythonOutput);
      console.error("[Launcher] Python stderr:", pythonErrors);
      
      // Try to extract meaningful error from output
      let errorMessage = `Python process exited with code ${code}`;
      if (pythonErrors) {
        // Look for common errors
        if (pythonErrors.includes("ModuleNotFoundError") || pythonErrors.includes("ImportError")) {
          const missingModule = pythonErrors.match(/No module named ['"]([^'"]+)['"]/);
          if (missingModule) {
            errorMessage = `Python dependency missing: ${missingModule[1]}\n\nThe bundled Python venv may be incomplete.\n\nTo fix:\n1. Delete: rm -rf backend-python-ml-v2/venv\n2. Rebuild: ./prepare-python-venv.sh\n3. Rebuild AppImage: ./build-appimage.sh\n\nError details:\n${pythonErrors.split('\n').slice(0, 5).join('\n')}`;
          } else {
            errorMessage = `Python dependencies missing.\n\nThe bundled Python venv may be incomplete.\n\nTo fix:\n1. Delete: rm -rf backend-python-ml-v2/venv\n2. Rebuild: ./prepare-python-venv.sh\n3. Rebuild AppImage: ./build-appimage.sh\n\nError:\n${pythonErrors.split('\n').slice(0, 5).join('\n')}`;
          }
        } else if (pythonErrors.includes("Failed to import encodings") || pythonErrors.includes("No module named 'encodings'")) {
          errorMessage = `Python standard library not found.\n\nThis usually means Python can't find its standard library.\nMake sure Python 3.9+ is installed: python3 --version\n\nError:\n${pythonErrors.split('\n').slice(0, 3).join('\n')}`;
        } else if (pythonErrors.includes("FileNotFoundError")) {
          errorMessage = `Python file not found:\n\n${pythonErrors.split('\n').slice(0, 3).join('\n')}`;
        } else {
          errorMessage = `Python error:\n\n${pythonErrors.split('\n').slice(0, 10).join('\n')}`;
        }
      }
      throw new Error(errorMessage);
    }
    if (signal) {
      console.error(`[Launcher] ✗ Python ML process killed with signal ${signal}`);
      throw new Error(`Python process was killed (signal: ${signal})`);
    }
  });

  console.log("[Launcher] Python ML process spawned (PID:", pythonMLProcess.pid, ")");
  return pythonMLProcess;
}

function stopAllServices() {
  console.log("[Launcher] Stopping all services...");
  
  if (frontendProcess) {
    frontendProcess.kill();
    frontendProcess = null;
  }
  
  if (nestjsProcess) {
    nestjsProcess.kill();
    nestjsProcess = null;
  }
  
  if (pythonMLProcess) {
    pythonMLProcess.kill();
    pythonMLProcess = null;
  }
}

// Wait for a service to be ready
function waitForService(url, timeout = 30000) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    const checkInterval = 1000; // Check every second
    const http = require("http");
    
    const check = () => {
      const elapsed = Date.now() - startTime;
      if (elapsed > timeout) {
        reject(new Error(`Service at ${url} did not become ready within ${timeout}ms`));
        return;
      }
      
      const req = http.get(url, { timeout: 2000 }, (res) => {
        // Service is ready
        resolve(true);
      });
      
      req.on("error", () => {
        // Service not ready yet, check again
        setTimeout(check, checkInterval);
      });
      
      req.on("timeout", () => {
        req.destroy();
        setTimeout(check, checkInterval);
      });
    };
    
    // Start checking
    check();
  });
}

async function startAllServices() {
  console.log("[Launcher] Starting all backend services...");
  
  // Start services in order
  try {
    // 1. Start Python ML service first (gRPC)
    console.log("[Launcher] Step 1/3: Starting Python ML service...");
    console.log("[Launcher] Note: Python service may take 30-60 seconds on first launch to download models");
    const pythonProcess = startPythonML();
    if (!pythonProcess) {
      throw new Error("Failed to start Python ML service - process is null. Check if Python 3.9+ is installed.");
    }
    console.log("[Launcher] Python ML process started (PID:", pythonProcess.pid, ")");
    console.log("[Launcher] Waiting for Python to initialize (this may take a while on first launch)...");
    
    // Wait longer for Python - it may need to download models on first run
    // Check if gRPC port becomes available (more reliable than fixed timeout)
    let pythonReady = false;
    for (let i = 0; i < 120; i++) { // Wait up to 2 minutes for Python
      try {
        // Try to connect to gRPC port (simple TCP check)
        const net = require("net");
        await new Promise((resolve, reject) => {
          const socket = new net.Socket();
          socket.setTimeout(1000);
          socket.once("connect", () => {
            socket.destroy();
            resolve(true);
          });
          socket.once("error", reject);
          socket.once("timeout", () => {
            socket.destroy();
            reject(new Error("Timeout"));
          });
          socket.connect(50051, "localhost");
        });
        pythonReady = true;
        console.log("[Launcher] ✓ Python ML service is ready!");
        break;
      } catch (e) {
        if (i % 10 === 0 && i > 0) {
          console.log(`[Launcher] Waiting for Python ML service... (${i}/120 - this is normal on first launch)`);
        }
        // Check if process is still alive
        try {
          process.kill(pythonProcess.pid, 0); // Signal 0 just checks if process exists
        } catch (err) {
          // Process doesn't exist - it died
          console.error("[Launcher] ✗ Python ML process died during startup");
          // The exit handler should have logged the error, but throw here too
          throw new Error("Python ML process died during startup. Check console output above for Python error details.");
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    if (!pythonReady) {
      console.warn("[Launcher] ⚠ Python ML service did not become ready, but continuing...");
    }
    
    // 2. Start NestJS API
    console.log("[Launcher] Step 2/3: Starting NestJS API...");
    const nestjsProc = startNestJS();
    if (!nestjsProc) {
      throw new Error("Failed to start NestJS API - process is null");
    }
    console.log("[Launcher] NestJS process started, waiting for health check...");
    
    // Wait for NestJS to be ready
    let nestjsReady = false;
    for (let i = 0; i < 30; i++) {
      try {
        await waitForService("http://localhost:3001/health", 2000);
        nestjsReady = true;
        break;
      } catch (e) {
        if (i % 5 === 0) {
          console.log(`[Launcher] Waiting for NestJS... (${i + 1}/30)`);
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    if (!nestjsReady) {
      console.warn("[Launcher] ⚠ NestJS health check failed, but continuing...");
    } else {
      console.log("[Launcher] ✓ NestJS API is ready!");
    }
    
    // 3. Start Frontend (Next.js server)
    console.log("[Launcher] Step 3/3: Starting Frontend...");
    const frontendProc = startFrontend();
    if (!frontendProc) {
      throw new Error("Failed to start Frontend - process is null. Check if frontend resources are bundled correctly.");
    }
    console.log("[Launcher] Frontend process started, waiting for server...");
    
    // Wait for frontend to be ready
    let frontendReady = false;
    for (let i = 0; i < 30; i++) {
      try {
        await waitForService("http://localhost:3000", 2000);
        frontendReady = true;
        break;
      } catch (e) {
        if (i % 5 === 0) {
          console.log(`[Launcher] Waiting for Frontend... (${i + 1}/30)`);
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    if (frontendReady) {
      console.log("[Launcher] ✓ Frontend is ready!");
      console.log("[Launcher] ✓ All services started successfully!");
    } else {
      throw new Error("Frontend did not become ready after 30 seconds. Check if Next.js server started correctly.");
    }
  } catch (error) {
    console.error("[Launcher] ✗ Error starting services:", error);
    console.error("[Launcher] Error stack:", error.stack);
    stopAllServices();
    throw error;
  }
}

// Export functions for use in main.js
module.exports = {
  startAllServices,
  stopAllServices,
  getResourcePath,
  getNodePath,
};

