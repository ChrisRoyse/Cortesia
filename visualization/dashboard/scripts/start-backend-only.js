#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const rootPath = path.resolve(__dirname, '../../..');
const dashboardPath = path.resolve(__dirname, '..');

console.log('🚀 Starting LLMKG Backend Services Only');
console.log('=======================================');

// Check if we're in the right directory
if (!fs.existsSync(path.join(rootPath, 'Cargo.toml'))) {
    console.error('❌ Error: Could not find Cargo.toml in parent directory');
    console.error('Make sure you\'re running this from the visualization/dashboard directory');
    process.exit(1);
}

let backendProcess = null;

// Cleanup function
async function cleanup() {
    console.log('\n🧹 Cleaning up...');
    
    if (backendProcess) {
        if (process.platform === 'win32') {
            spawn('taskkill', ['/F', '/T', '/PID', backendProcess.pid]);
        } else {
            backendProcess.kill('SIGTERM');
        }
    }
    
    process.exit(0);
}

// Handle termination signals
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
process.on('exit', cleanup);

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
                console.log('\n✅ Backend server launched!');
                console.log('\n📡 Services available at:');
                console.log('   API:       http://localhost:3001/api/v1');
                console.log('   Dashboard: http://localhost:8090');
                console.log('   WebSocket: ws://localhost:8081');
                console.log('\n📊 Open http://localhost:8090 in your browser to view the dashboard');
                console.log('\nPress Ctrl+C to stop the server');
                resolve();
            }, 5000);
        });
    });
}

async function main() {
    try {
        await startBackend();
    } catch (error) {
        console.error('❌ Failed to start backend:', error.message);
        process.exit(1);
    }
}

main();