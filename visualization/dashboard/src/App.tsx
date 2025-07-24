import React, { Suspense } from 'react';
import { Provider } from 'react-redux';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { store } from './stores';
import { WebSocketProvider } from './providers/WebSocketProvider';
import { MCPProvider, MCPProviderDev } from './providers/MCPProvider';
import { ThemeProvider } from './components/ThemeProvider/ThemeProvider';
// import { DashboardLayout } from './components/Layout/DashboardLayout';
import './styles/globals.css';

// Lazy load pages for code splitting
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const CognitivePage = React.lazy(() => import('./pages/CognitivePage'));
const NeuralPage = React.lazy(() => import('./pages/NeuralPage'));
const KnowledgeGraphPage = React.lazy(() => import('./pages/KnowledgeGraphPage'));
const RealKnowledgeGraphPage = React.lazy(() => import('./pages/RealKnowledgeGraphPage'));
const MemoryPage = React.lazy(() => import('./pages/MemoryPage'));
const DebuggingPage = React.lazy(() => import('./pages/DebuggingPage'));
const ToolsPage = React.lazy(() => import('./pages/ToolsPage'));
const SettingsPage = React.lazy(() => import('./pages/SettingsPage'));
const ArchitecturePage = React.lazy(() => import('./pages/Architecture/ArchitecturePage'));

// Configuration
const isDevelopment = import.meta.env.DEV;
const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8081';
const MCP_SERVER_URL = import.meta.env.VITE_MCP_SERVER_URL || 'http://localhost:3000';

// Loading component
const LoadingSpinner: React.FC = () => (
  <div className="loading-container">
    <div className="loading-spinner" />
    <div className="loading-text">Loading...</div>
  </div>
);

// Error fallback component
interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({ error, resetErrorBoundary }) => (
  <div className="error-container">
    <div className="error-content">
      <h2 className="error-title">Something went wrong</h2>
      <details className="error-details">
        <summary>Error details</summary>
        <pre className="error-message">{error.message}</pre>
        <pre className="error-stack">{error.stack}</pre>
      </details>
      <button className="error-button" onClick={resetErrorBoundary}>
        Try again
      </button>
    </div>
  </div>
);

// Custom error boundary component
class AppErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  override componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('App Error Boundary caught an error:', error, errorInfo);
    
    // In production, you might want to send this to an error reporting service
    if (!isDevelopment) {
      // Example: Sentry.captureException(error);
    }
  }

  override render() {
    if (this.state.hasError && this.state.error) {
      return (
        <ErrorFallback
          error={this.state.error}
          resetErrorBoundary={() => this.setState({ hasError: false, error: null })}
        />
      );
    }

    return this.props.children;
  }
}

// Route error boundary for individual pages
const RouteErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AppErrorBoundary>
    <Suspense fallback={<LoadingSpinner />}>
      {children}
    </Suspense>
  </AppErrorBoundary>
);

// Main App Routes
const AppRoutes: React.FC = () => (
  <Routes>
    <Route 
      path="/" 
      element={
        <RouteErrorBoundary>
          <Dashboard />
        </RouteErrorBoundary>
      } 
    />
    <Route 
      path="/cognitive" 
      element={
        <RouteErrorBoundary>
          <CognitivePage />
        </RouteErrorBoundary>
      } 
    />
    <Route 
      path="/neural" 
      element={
        <RouteErrorBoundary>
          <NeuralPage />
        </RouteErrorBoundary>
      } 
    />
    <Route 
      path="/knowledge-graph" 
      element={
        <RouteErrorBoundary>
          <KnowledgeGraphPage />
        </RouteErrorBoundary>
      } 
    />
    <Route 
      path="/real-knowledge-graph" 
      element={
        <RouteErrorBoundary>
          <RealKnowledgeGraphPage />
        </RouteErrorBoundary>
      } 
    />
    <Route 
      path="/memory" 
      element={
        <RouteErrorBoundary>
          <MemoryPage />
        </RouteErrorBoundary>
      } 
    />
    <Route 
      path="/debugging" 
      element={
        <RouteErrorBoundary>
          <DebuggingPage />
        </RouteErrorBoundary>
      } 
    />
    <Route 
      path="/tools" 
      element={
        <RouteErrorBoundary>
          <ToolsPage />
        </RouteErrorBoundary>
      } 
    />
    <Route 
      path="/settings" 
      element={
        <RouteErrorBoundary>
          <SettingsPage />
        </RouteErrorBoundary>
      } 
    />
    <Route 
      path="/architecture" 
      element={
        <RouteErrorBoundary>
          <ArchitecturePage />
        </RouteErrorBoundary>
      } 
    />
    {/* Catch all route - redirect to home */}
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes>
);

// Main App Component
const App: React.FC = () => {
  const MCPComponent = isDevelopment ? MCPProviderDev : MCPProvider;
  
  return (
    <AppErrorBoundary>
      <Provider store={store}>
        <ThemeProvider 
          defaultMode="dark"
          enableSystemPreference={true}
          storageKey="llmkg-theme-mode"
        >
          <WebSocketProvider 
            url={WEBSOCKET_URL}
            reconnectDelay={3000}
            heartbeatInterval={30000}
          >
            <MCPComponent serverUrl={MCP_SERVER_URL}>
              <Router future={{
                v7_startTransition: true,
                v7_relativeSplatPath: true
              }}>
                <AppRoutes />
              </Router>
            </MCPComponent>
          </WebSocketProvider>
        </ThemeProvider>
      </Provider>
    </AppErrorBoundary>
  );
};

export default App;

// Performance monitoring (development only)
if (isDevelopment && typeof window !== 'undefined') {
  // Monitor performance metrics
  const observer = new PerformanceObserver((list) => {
    list.getEntries().forEach((entry) => {
      console.log(`Performance: ${entry.name} - ${entry.duration}ms`);
    });
  });
  
  observer.observe({ entryTypes: ['navigation', 'measure'] });
  
  // Monitor WebSocket connection performance
  window.addEventListener('llmkg:websocket:connected', () => {
    performance.mark('websocket-connected');
  });
  
  window.addEventListener('llmkg:data:received', () => {
    performance.mark('data-received');
  });
}