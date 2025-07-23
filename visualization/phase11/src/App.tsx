import React, { Suspense, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Spin, Alert } from 'antd';
import { motion, AnimatePresence } from 'framer-motion';

import { LLMKGVisualizationProvider } from './services/VisualizationCore';
import { AppLayout } from './components/Layout/AppLayout';
import { LoadingScreen } from './components/Loading/LoadingScreen';
import { useAppConfig } from './hooks/useAppConfig';
import { usePerformanceMonitoring } from './hooks/usePerformanceMonitoring';

// Lazy load major components for better performance
const SystemOverview = React.lazy(() => import('./components/Dashboard/SystemOverview'));
const MemoryDashboard = React.lazy(() => import('./components/Memory/MemoryDashboard'));
const CognitiveDashboard = React.lazy(() => import('./components/Cognitive/CognitiveDashboard'));
const DebuggingDashboard = React.lazy(() => import('./components/Debug/DebuggingDashboard'));
const PerformanceMonitor = React.lazy(() => import('./components/Performance/PerformanceMonitor'));
const ComponentRegistry = React.lazy(() => import('./components/Registry/ComponentRegistry'));
const VersionControl = React.lazy(() => import('./components/Version/VersionControl'));
const DocumentationHub = React.lazy(() => import('./components/Documentation/DocumentationHub'));
const SettingsPage = React.lazy(() => import('./components/Settings/SettingsPage'));

// Page transition variants
const pageVariants = {
  initial: { opacity: 0, y: 20 },
  in: { opacity: 1, y: 0 },
  out: { opacity: 0, y: -20 }
};

const pageTransition = {
  type: 'tween',
  ease: 'anticipate',
  duration: 0.3
};

// Loading fallback component
const PageLoadingFallback: React.FC = () => (
  <div style={{ 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    height: '60vh',
    flexDirection: 'column',
    gap: '16px'
  }}>
    <Spin size="large" />
    <span style={{ color: 'rgba(255, 255, 255, 0.65)' }}>Loading component...</span>
  </div>
);

// Error fallback for route-level errors
const RouteErrorFallback: React.FC<{ error?: Error }> = ({ error }) => (
  <div style={{ padding: '24px' }}>
    <Alert
      message="Component Load Error"
      description={error?.message || 'Failed to load this page component. Please try refreshing.'}
      type="error"
      showIcon
      action={
        <button 
          onClick={() => window.location.reload()}
          style={{
            border: 'none',
            background: '#ff4d4f',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Refresh
        </button>
      }
    />
  </div>
);

const App: React.FC = () => {
  const config = useAppConfig();
  const { startMonitoring } = usePerformanceMonitoring();

  useEffect(() => {
    // Start performance monitoring
    startMonitoring();
    
    // Set up global error handling
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', event.reason);
      // In production, this would be sent to error monitoring service
    };

    const handleError = (event: ErrorEvent) => {
      console.error('Global error:', event.error);
      // In production, this would be sent to error monitoring service
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    window.addEventListener('error', handleError);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      window.removeEventListener('error', handleError);
    };
  }, [startMonitoring]);

  if (!config) {
    return <LoadingScreen message="Initializing LLMKG Visualization System..." />;
  }

  return (
    <LLMKGVisualizationProvider config={config}>
      <AppLayout>
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<Navigate to="/overview" replace />} />
            
            <Route 
              path="/overview" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <Suspense fallback={<PageLoadingFallback />}>
                    <SystemOverview />
                  </Suspense>
                </motion.div>
              } 
            />
            
            <Route 
              path="/memory" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <Suspense fallback={<PageLoadingFallback />}>
                    <MemoryDashboard />
                  </Suspense>
                </motion.div>
              } 
            />
            
            <Route 
              path="/cognitive" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <Suspense fallback={<PageLoadingFallback />}>
                    <CognitiveDashboard />
                  </Suspense>
                </motion.div>
              } 
            />
            
            <Route 
              path="/debugging" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <Suspense fallback={<PageLoadingFallback />}>
                    <DebuggingDashboard />
                  </Suspense>
                </motion.div>
              } 
            />
            
            <Route 
              path="/performance" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <Suspense fallback={<PageLoadingFallback />}>
                    <PerformanceMonitor />
                  </Suspense>
                </motion.div>
              } 
            />
            
            <Route 
              path="/registry" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <Suspense fallback={<PageLoadingFallback />}>
                    <ComponentRegistry />
                  </Suspense>
                </motion.div>
              } 
            />
            
            <Route 
              path="/version" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <Suspense fallback={<PageLoadingFallback />}>
                    <VersionControl />
                  </Suspense>
                </motion.div>
              } 
            />
            
            <Route 
              path="/docs" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <Suspense fallback={<PageLoadingFallback />}>
                    <DocumentationHub />
                  </Suspense>
                </motion.div>
              } 
            />
            
            <Route 
              path="/settings" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <Suspense fallback={<PageLoadingFallback />}>
                    <SettingsPage />
                  </Suspense>
                </motion.div>
              } 
            />
            
            {/* Catch-all route for 404s */}
            <Route 
              path="*" 
              element={
                <motion.div
                  initial="initial"
                  animate="in"
                  exit="out"
                  variants={pageVariants}
                  transition={pageTransition}
                >
                  <RouteErrorFallback error={new Error('Page not found')} />
                </motion.div>
              } 
            />
          </Routes>
        </AnimatePresence>
      </AppLayout>
    </LLMKGVisualizationProvider>
  );
};

export default App;