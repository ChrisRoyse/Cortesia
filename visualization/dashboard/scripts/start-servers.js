#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const rootPath = path.resolve(__dirname, '../../..');
const dashboardPath = path.resolve(__dirname, '..');

console.log('🚀 Starting LLMKG Development Environment');
console.log('========================================');

// Check if we're in the right directory
if (!fs.existsSync(path.join(rootPath, 'Cargo.toml'))) {
    console.error('❌ Error: Could not find Cargo.toml in parent directory');
    console.error('Make sure you\'re running this from the visualization/dashboard directory');
    process.exit(1);
}

// Initial cleanup function to kill existing processes
async function initialCleanup() {
    console.log('🧹 Cleaning up existing processes...');
    
    if (process.platform === 'win32') {
        try {
            const { execSync } = require('child_process');
            
            // Only kill the specific backend process
            try {
                execSync(`taskkill /F /IM llmkg_api_server.exe 2>nul`, { stdio: 'ignore' });
                console.log(`✅ Killed existing llmkg_api_server.exe processes`);
            } catch (e) {
                // Process not found, which is fine
            }
            
            // Kill processes using specific ports - but check if they're our processes first
            const portsToFree = [3001, 5173, 8081];
            for (const port of portsToFree) {
                try {
                    const pidOutput = execSync(`netstat -ano | findstr :${port}`, { encoding: 'utf8', stdio: 'pipe' });
                    const lines = pidOutput.split('\n').filter(line => line.trim());
                    
                    for (const line of lines) {
                        // Only match LISTENING state to avoid killing clients
                        if (!line.includes('LISTENING')) continue;
                        
                        const match = line.match(/\s+(\d+)\s*$/);
                        if (match) {
                            const pid = match[1];
                            try {
                                // Get process info to check if it's our process
                                const processInfo = execSync(`wmic process where ProcessId=${pid} get Name,CommandLine`, { encoding: 'utf8' });
                                
                                // Only kill if it's vite, cargo, or llmkg related
                                if (processInfo.includes('vite') || 
                                    processInfo.includes('llmkg') || 
                                    processInfo.includes('cargo') ||
                                    (port === 5173 && processInfo.includes('node'))) { // Only kill node on port 5173
                                    
                                    execSync(`taskkill /F /PID ${pid}`, { stdio: 'ignore' });
                                    console.log(`✅ Freed port ${port} (PID: ${pid})`);
                                }
                            } catch (e) {
                                // PID might not exist anymore or wmic failed
                            }
                        }
                    }
                } catch (e) {
                    // Port not in use, which is fine
                }
            }
            
            // Wait longer for processes to fully terminate
            console.log('⏳ Waiting for processes to terminate...');
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Clean up locked files more thoroughly
            const pathsToClean = [
                path.join(rootPath, 'target', 'debug', 'llmkg_api_server.exe'),
                path.join(rootPath, 'target', 'debug', 'llmkg_api_server.pdb'),
                path.join(rootPath, 'target', 'debug', 'deps', 'llmkg_api_server.exe'),
            ];
            
            for (const filePath of pathsToClean) {
                try {
                    if (fs.existsSync(filePath)) {
                        // Try multiple times with increasing delays
                        let attempts = 0;
                        const maxAttempts = 3;
                        
                        while (attempts < maxAttempts) {
                            try {
                                fs.unlinkSync(filePath);
                                console.log(`✅ Cleaned up ${path.basename(filePath)}`);
                                break;
                            } catch (e) {
                                attempts++;
                                if (attempts < maxAttempts) {
                                    console.log(`🔄 Retrying cleanup of ${path.basename(filePath)} (attempt ${attempts + 1}/${maxAttempts})`);
                                    await new Promise(resolve => setTimeout(resolve, 1000));
                                }
                            }
                        }
                        
                        // If still locked, try cargo clean as last resort
                        if (attempts === maxAttempts && fs.existsSync(filePath)) {
                            console.log('🔧 File still locked, trying cargo clean...');
                            try {
                                execSync('cargo clean --bin llmkg_api_server', { cwd: rootPath, stdio: 'inherit' });
                                console.log('✅ Cargo clean completed');
                            } catch (cleanError) {
                                console.log('⚠️  Warning: Could not clean binary, build may take longer');
                            }
                        }
                    }
                } catch (error) {
                    console.log(`⚠️  Warning: Could not clean ${path.basename(filePath)}: ${error.message}`);
                }
            }
            
        } catch (error) {
            console.log('⚠️  Warning: Cleanup had issues, continuing anyway...');
        }
    } else {
        // Unix-like systems cleanup
        try {
            const { execSync } = require('child_process');
            
            // Kill processes on specific ports
            const portsToFree = [3001, 5173, 8081];
            for (const port of portsToFree) {
                try {
                    execSync(`lsof -ti:${port} | xargs kill -9`, { stdio: 'ignore' });
                    console.log(`✅ Freed port ${port}`);
                } catch (e) {
                    // Port not in use, which is fine
                }
            }
            
            await new Promise(resolve => setTimeout(resolve, 1000));
            
        } catch (error) {
            console.log('⚠️  Warning: Unix cleanup had issues, continuing anyway...');
        }
    }
}

let backendProcess = null;
let frontendProcess = null;

// Cleanup function
function cleanup() {
    console.log('\n🛑 Shutting down servers...');
    if (backendProcess) {
        backendProcess.kill('SIGTERM');
    }
    if (frontendProcess) {
        frontendProcess.kill('SIGTERM');
    }
    process.exit(0);
}

// Handle shutdown signals
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

async function startBackend() {
    return new Promise((resolve, reject) => {
        console.log('📦 Building LLMKG API server...');
        
        const buildProcess = spawn('cargo', ['build', '--bin', 'llmkg_api_server'], {
            cwd: rootPath,
            stdio: 'inherit'
        });

        buildProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Backend build failed with code ${code}`));
                return;
            }

            console.log('✅ Backend built successfully');
            console.log('🚀 Starting LLMKG API server...');

            backendProcess = spawn('cargo', ['run', '--bin', 'llmkg_api_server'], {
                cwd: rootPath,
                stdio: 'inherit'
            });

            backendProcess.on('error', (err) => {
                console.error('❌ Backend process error:', err);
                reject(err);
            });

            // Give the server time to start
            setTimeout(() => {
                console.log('✅ Backend server process launched');
                resolve();
            }, 5000);
        });
    });
}

async function waitForBackend() {
    const maxAttempts = 60; // Increased from 30 to 60 seconds
    let attempts = 0;
    let apiReady = false;
    let wsReady = false;

    console.log('⏳ Waiting for backend servers to be ready...');
    console.log('   This may take a minute if the code is being compiled...');

    while (attempts < maxAttempts) {
        // Check API server
        if (!apiReady) {
            try {
                const apiResponse = await fetch('http://localhost:3001/api/v1/discovery');
                if (apiResponse.ok) {
                    apiReady = true;
                    console.log('✅ API server is ready!');
                }
            } catch (error) {
                // API not ready yet
            }
        }

        // Check WebSocket
        if (!wsReady) {
            try {
                const ws = new (require('ws'))('ws://localhost:8081');
                await new Promise((resolve, reject) => {
                    ws.on('open', () => {
                        ws.close();
                        wsReady = true;
                        console.log('✅ WebSocket server is ready!');
                        resolve();
                    });
                    ws.on('error', reject);
                    setTimeout(() => reject(new Error('WebSocket timeout')), 1000);
                });
            } catch (error) {
                // WebSocket not ready yet
            }
        }

        // If both are ready, we're done
        if (apiReady && wsReady) {
            console.log('✅ All backend services are ready!');
            return;
        }

        attempts++;
        if (attempts % 10 === 0) {
            console.log(`   Still waiting... (${attempts}/${maxAttempts} seconds)`);
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Provide helpful error message
    let errorMsg = 'Backend services failed to start within timeout.\n';
    if (!apiReady) errorMsg += '  - API server (port 3001) is not responding\n';
    if (!wsReady) errorMsg += '  - WebSocket server (port 8081) is not responding\n';
    errorMsg += '\nTroubleshooting:\n';
    errorMsg += '  1. Check if the Rust compilation completed successfully\n';
    errorMsg += '  2. Look for error messages in the terminal above\n';
    errorMsg += '  3. Try running "cargo build --bin llmkg_api_server" manually\n';
    
    throw new Error(errorMsg);
}

async function startFrontend() {
    console.log('🎨 Starting Vite development server...');

    frontendProcess = spawn(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'dev:frontend-only'], {
        cwd: dashboardPath,
        stdio: 'inherit'
    });

    frontendProcess.on('error', (err) => {
        console.error('❌ Frontend process error:', err);
    });
}

async function main() {
    try {
        await initialCleanup();
        await startBackend();
        await waitForBackend();
        await startFrontend();

        console.log('\n🎉 All servers started successfully!');
        console.log('📡 API endpoints: http://localhost:3001/api/v1');
        console.log('📊 Dashboard: http://localhost:8090');
        console.log('🔌 WebSocket: ws://localhost:8081');
        console.log('🎨 Vite dev server: http://localhost:5173');
        console.log('\nPress Ctrl+C to stop all servers');

    } catch (error) {
        console.error('❌ Failed to start servers:', error.message);
        cleanup();
    }
}

main();