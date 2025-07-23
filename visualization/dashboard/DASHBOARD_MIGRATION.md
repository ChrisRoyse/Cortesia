# Dashboard Migration Notice

The visualization dashboard has been migrated to use the backend monitoring dashboard.

## Current Setup:
- Main Dashboard: http://localhost:8090 (served by Rust backend)
- API Endpoints: http://localhost:3001/api/v1
- WebSocket: ws://localhost:8081

## To Run:
```bash
npm run dev
```

This will start only the backend services. The Vite dev server (port 5173) has been disabled.
