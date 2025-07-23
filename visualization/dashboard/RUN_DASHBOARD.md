# Running the LLMKG Dashboard

This guide explains how to run the unified LLMKG visualization dashboard.

## Prerequisites

1. **Build the LLMKG Server**
   ```bash
   cd C:\code\LLMKG
   cargo build --bin llmkg-server --release
   ```

2. **Install Dashboard Dependencies**
   ```bash
   cd visualization/dashboard
   npm install --legacy-peer-deps
   ```

## Running the System

### Step 1: Start the LLMKG Server

The server provides real-time data via WebSocket on port 8081:

```bash
# From the LLMKG root directory
cargo run --bin llmkg-server --release
```

This will start:
- HTTP Dashboard: http://localhost:8080
- WebSocket: ws://localhost:8081

### Step 2: Start the Dashboard

In a new terminal:

```bash
cd visualization/dashboard
npm run dev
```

The dashboard will be available at http://localhost:3001 (or the next available port).

## Features

The unified dashboard includes:
- **Phase 7**: Memory monitoring and consolidation
- **Phase 8**: Cognitive pattern visualization
- **Phase 9**: Debugging tools and tracing
- **Phase 10**: System integration layer
- **Phase 11**: Performance optimizations

## Troubleshooting

### Port Conflicts
If you see "Port 3001 is in use", the dashboard will automatically try the next available port.

### WebSocket Connection Issues
Make sure the LLMKG server is running before starting the dashboard. The WebSocket connects to `ws://localhost:8081`.

### Build Errors
If you encounter build errors, try:
```bash
cargo clean
cargo build --bin llmkg-server --release
```

## Architecture

The system consists of:
1. **LLMKG Server** (Rust): Provides real-time metrics and data via WebSocket
2. **Dashboard** (React/TypeScript): Visualizes the data with advanced UI components

The dashboard uses:
- React 18 with TypeScript
- Redux Toolkit for state management
- Material-UI and Ant Design for UI components
- Three.js for 3D visualizations
- D3.js for data visualizations
- WebSocket for real-time updates