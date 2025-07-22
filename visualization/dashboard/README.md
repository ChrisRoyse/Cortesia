# LLMKG Dashboard

Real-time visualization dashboard for Large Language Model Knowledge Graphs. This is Phase 2 of the LLMKG visualization system, providing an interactive React-based interface for monitoring cognitive patterns, neural activity, knowledge graphs, and memory systems.

## Features

### Core Architecture
- **React 18** with concurrent features for optimal performance
- **TypeScript** with strict typing for production-ready code
- **Redux Toolkit** for state management
- **React Router** for multi-page navigation
- **WebSocket** integration for real-time data streaming
- **MCP (Model Control Protocol)** integration for tool execution

### Real-Time Capabilities
- Live data streaming from WebSocket server (ws://localhost:8080)
- Automatic reconnection with exponential backoff
- Real-time cognitive pattern visualization
- Neural activity monitoring
- Knowledge graph updates
- Memory system performance tracking

### Dashboard Views
- **Overview** - System status and key metrics
- **Cognitive** - Cognitive patterns and inhibitory mechanisms
- **Neural** - Neural network activity and connections
- **Knowledge Graph** - Interactive graph visualization
- **Memory** - Memory usage and performance metrics
- **Tools** - MCP tool catalog and execution interface
- **Settings** - Dashboard configuration and preferences

### Performance Optimizations
- Code splitting with React.lazy()
- Virtual scrolling for large datasets
- Efficient re-rendering with React.memo and useMemo
- <100ms UI update latency from WebSocket data
- 60 FPS animations and transitions

## Quick Start

### Prerequisites
- Node.js >= 18.0.0
- npm >= 8.0.0
- Phase 1 WebSocket server running on ws://localhost:8080

### Installation
```bash
cd visualization/dashboard
npm install
```

### Development
```bash
npm start
```
Open http://localhost:3000 in your browser.

### Production Build
```bash
npm run build
npm run serve
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   └── Layout/         # Layout components
├── pages/              # Page components
├── providers/          # React context providers
│   ├── WebSocketProvider.tsx
│   └── MCPProvider.tsx
├── stores/             # Redux store and slices
├── types/              # TypeScript type definitions
├── App.tsx             # Main application component
├── index.tsx           # Entry point
└── index.css           # Global styles
```

## Architecture

### State Management
The dashboard uses Redux Toolkit for centralized state management with the following slices:
- **dashboard** - UI state, theme, settings
- **data** - Real-time LLMKG data with history
- **webSocket** - Connection state and message handling
- **mcp** - MCP tools and execution history

### WebSocket Integration
Real-time data flows through the WebSocketProvider:
- Automatic connection management
- Message parsing and validation
- Topic-based subscriptions
- Error handling and reconnection
- Performance monitoring

### MCP Integration
Model Control Protocol integration via MCPProvider:
- Tool discovery and cataloging
- Execution tracking with history
- Error handling and retries
- Batch execution support

## Configuration

### Environment Variables
Create `.env` file for configuration:
```bash
REACT_APP_WEBSOCKET_URL=ws://localhost:8080
REACT_APP_MCP_SERVER_URL=http://localhost:3000
REACT_APP_VERSION=2.0.0
```

### Theme Configuration
The dashboard supports multiple themes:
- **Dark** (default) - Optimized for extended use
- **Light** - High contrast for accessibility
- **Auto** - Follows system preference

### Performance Settings
Configurable performance parameters:
- Refresh rate (500ms - 5000ms)
- Max data points (500 - 5000)
- Animation toggle
- WebSocket reconnection settings

## Integration with Phase 1

The dashboard integrates seamlessly with Phase 1 infrastructure:

### WebSocket Data Streams
Expects structured data from Phase 1 server:
```typescript
interface LLMKGData {
  cognitive: CognitiveData;
  neural: NeuralData;
  knowledgeGraph: KnowledgeGraphData;
  memory: MemoryData;
  timestamp: number;
}
```

### MCP Tool Integration
Connects to Phase 1 MCP server for:
- Tool discovery and execution
- Real-time system interaction
- Performance testing and monitoring

## Development

### Code Quality
- ESLint for code linting
- TypeScript strict mode
- React Hooks rules
- Accessibility compliance (WCAG 2.1)

### Testing
```bash
npm test
```

### Type Checking
```bash
npm run type-check
```

### Bundle Analysis
```bash
npm run analyze
```

## Performance Considerations

### Optimization Strategies
1. **React.memo** for component memoization
2. **useMemo** for expensive calculations
3. **useCallback** for stable function references
4. **Code splitting** for reduced initial bundle size
5. **Virtual scrolling** for large data sets

### Monitoring
Built-in performance monitoring:
- Web Vitals reporting
- WebSocket connection metrics
- Rendering performance tracking
- Memory usage monitoring

## Accessibility

### Features
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support
- Reduced motion preferences
- ARIA labels and descriptions

### Testing
- Automated accessibility testing
- Manual keyboard testing
- Screen reader testing
- Color contrast validation

## Browser Support

### Supported Browsers
- Chrome/Chromium >= 90
- Firefox >= 88
- Safari >= 14
- Edge >= 90

### Features Used
- ES2020 features
- WebSockets
- CSS Grid and Flexbox
- CSS Custom Properties
- Performance Observer API

## Deployment

### Production Build
```bash
npm run build
```

### Serving
```bash
npm run serve
```

### Docker (Optional)
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "serve"]
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Verify Phase 1 server is running
   - Check firewall settings
   - Confirm WebSocket URL configuration

2. **MCP Tools Not Loading**
   - Verify MCP server is accessible
   - Check CORS configuration
   - Review server logs for errors

3. **Performance Issues**
   - Reduce refresh rate in settings
   - Lower max data points
   - Disable animations
   - Check browser DevTools performance tab

### Debug Mode
Set `NODE_ENV=development` for:
- Detailed console logging
- Performance monitoring
- WebSocket connection debugging
- Redux DevTools integration

## Contributing

### Development Workflow
1. Follow TypeScript strict mode
2. Use ESLint configuration
3. Write comprehensive type definitions
4. Test WebSocket integration
5. Verify accessibility compliance

### Code Standards
- Functional components with hooks
- TypeScript for all new code
- Consistent naming conventions
- Comprehensive error handling
- Performance-conscious implementations

## License

This project is part of the LLMKG system and follows the same licensing terms.