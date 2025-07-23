#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const rootPath = path.resolve(__dirname, '../../..');
const dashboardPath = path.resolve(__dirname, '..');

console.log('🚀 Starting LLMKG Development Environment (Safe Mode)');
console.log('=====================================================');

// Check if we're in the right directory
if (!fs.existsSync(path.join(rootPath, 'Cargo.toml'))) {
    console.error('❌ Error: Could not find Cargo.toml in parent directory');
    console.error('Make sure you\'re running this from the visualization/dashboard directory');
    process.exit(1);
}

// Check if ports are available without killing processes
async function checkPorts() {
    const { execSync } = require('child_process');
    const portsToCheck = [
        { port: 3001, name: 'API Server' },
        { port: 5173, name: 'Vite Dev Server' },
        { port: 8081, name: 'WebSocket Server' }
    ];
    
    let allPortsFree = true;
    
    for (const { port, name } of portsToCheck) {
        try {
            if (process.platform === 'win32') {
                execSync(`netstat -ano | findstr :${port} | findstr LISTENING`, { stdio: 'pipe' });
                console.log(`⚠️  Port ${port} (${name}) is already in use`);
                allPortsFree = false;
            } else {
                execSync(`lsof -ti:${port}`, { stdio: 'pipe' });
                console.log(`⚠️  Port ${port} (${name}) is already in use`);
                allPortsFree = false;
            }
        } catch (e) {
            // Port is free
            console.log(`✅ Port ${port} (${name}) is available`);
        }
    }
    
    if (!allPortsFree) {
        console.log('\n⚠️  Some ports are already in use.');
        console.log('Options:');
        console.log('1. Stop the processes using these ports manually');
        console.log('2. Use "npm run dev:force" to forcefully kill processes (may affect other applications)');
        console.log('3. Configure different ports in your application');
        process.exit(1);
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
                console.log('✅ Backend server started');
                resolve();
            }, 3000);
        });
    });
}

async function waitForBackend() {
    const maxAttempts = 30;
    let attempts = 0;

    console.log('⏳ Waiting for backend servers to be ready...');

    while (attempts < maxAttempts) {
        try {
            // Check API server
            const apiResponse = await fetch('http://localhost:3001/api/v1/discovery');
            if (apiResponse.ok) {
                // Check WebSocket by attempting connection
                const ws = new (require('ws'))('ws://localhost:8081');
                await new Promise((resolve, reject) => {
                    ws.on('open', () => {
                        ws.close();
                        resolve();
                    });
                    ws.on('error', reject);
                    setTimeout(() => reject(new Error('WebSocket timeout')), 1000);
                });

                console.log('✅ All backend services are ready!');
                return;
            }
        } catch (error) {
            // Services not ready yet
        }

        attempts++;
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    throw new Error('Backend services failed to start within timeout');
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
        await checkPorts();
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