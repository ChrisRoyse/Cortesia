import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorBoundary } from '../components/common/ErrorBoundary';
import { DashboardLayout } from '../components/Layout/DashboardLayout';

// Lazy load pages for optimal performance
const DashboardPage = lazy(() => import('../pages/Dashboard/DashboardPage'));
const CognitivePage = lazy(() => import('../pages/Cognitive/CognitivePage'));
const NeuralPage = lazy(() => import('../pages/Neural/NeuralPage'));
const KnowledgeGraphPage = lazy(() => import('../pages/KnowledgeGraph/KnowledgeGraphPage'));
const MemoryPage = lazy(() => import('../pages/Memory/MemoryPage'));
const ToolsPage = lazy(() => import('../pages/Tools/ToolsPage'));
const ArchitecturePage = lazy(() => import('../pages/Architecture/ArchitecturePage'));

// Cognitive subsections
const PatternRecognitionPage = lazy(() => import('../pages/Cognitive/PatternRecognitionPage'));
const InhibitoryMechanismsPage = lazy(() => import('../pages/Cognitive/InhibitoryMechanismsPage'));
const AttentionSystemPage = lazy(() => import('../pages/Cognitive/AttentionSystemPage'));

// Neural subsections
const ActivityHeatmapPage = lazy(() => import('../pages/Neural/ActivityHeatmapPage'));
const ConnectivityMapPage = lazy(() => import('../pages/Neural/ConnectivityMapPage'));
const SpikeAnalysisPage = lazy(() => import('../pages/Neural/SpikeAnalysisPage'));

// Memory subsections
const PerformanceMetricsPage = lazy(() => import('../pages/Memory/PerformanceMetricsPage'));
const ConsolidationMonitorPage = lazy(() => import('../pages/Memory/ConsolidationMonitorPage'));
const UsageAnalyticsPage = lazy(() => import('../pages/Memory/UsageAnalyticsPage'));

// Tools subsections
const ToolCatalogPage = lazy(() => import('../pages/Tools/ToolCatalogPage'));
const ToolTestingPage = lazy(() => import('../pages/Tools/ToolTestingPage'));
const ExecutionHistoryPage = lazy(() => import('../pages/Tools/ExecutionHistoryPage'));

interface RouteGuardProps {
  children: React.ReactNode;
  requiresAuth?: boolean;
  requiredPermissions?: string[];
}

const RouteGuard: React.FC<RouteGuardProps> = ({ 
  children, 
  requiresAuth = false, 
  requiredPermissions = [] 
}) => {
  // TODO: Implement authentication and permission checks
  // For now, allow all access
  return <>{children}</>;
};

const PageLoader: React.FC = () => (
  <div className="flex items-center justify-center h-full min-h-[400px]">
    <LoadingSpinner size="large" message="Loading page..." />
  </div>
);

export const AppRouter: React.FC = () => {
  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={
            <RouteGuard>
              <Suspense fallback={<PageLoader />}>
                <DashboardPage />
              </Suspense>
            </RouteGuard>
          } />
          
          {/* Cognitive Routes */}
          <Route path="cognitive">
            <Route index element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <CognitivePage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="patterns" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <PatternRecognitionPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="inhibitory" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <InhibitoryMechanismsPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="attention" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <AttentionSystemPage />
                </Suspense>
              </RouteGuard>
            } />
          </Route>

          {/* Neural Routes */}
          <Route path="neural">
            <Route index element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <NeuralPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="activity" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <ActivityHeatmapPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="connectivity" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <ConnectivityMapPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="spikes" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <SpikeAnalysisPage />
                </Suspense>
              </RouteGuard>
            } />
          </Route>

          {/* Knowledge Graph Routes */}
          <Route path="knowledge-graph" element={
            <RouteGuard>
              <Suspense fallback={<PageLoader />}>
                <KnowledgeGraphPage />
              </Suspense>
            </RouteGuard>
          } />

          {/* Memory Routes */}
          <Route path="memory">
            <Route index element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <MemoryPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="performance" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <PerformanceMetricsPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="consolidation" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <ConsolidationMonitorPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="usage" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <UsageAnalyticsPage />
                </Suspense>
              </RouteGuard>
            } />
          </Route>

          {/* Tools Routes */}
          <Route path="tools">
            <Route index element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <ToolsPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="catalog" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <ToolCatalogPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="testing" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <ToolTestingPage />
                </Suspense>
              </RouteGuard>
            } />
            <Route path="history" element={
              <RouteGuard>
                <Suspense fallback={<PageLoader />}>
                  <ExecutionHistoryPage />
                </Suspense>
              </RouteGuard>
            } />
          </Route>

          {/* Architecture Routes */}
          <Route path="architecture" element={
            <RouteGuard requiredPermissions={['admin']}>
              <Suspense fallback={<PageLoader />}>
                <ArchitecturePage />
              </Suspense>
            </RouteGuard>
          } />

          {/* Catch all - redirect to dashboard */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </ErrorBoundary>
  );
};

// Route configuration for programmatic access
export const routeConfig = {
  dashboard: '/',
  cognitive: {
    index: '/cognitive',
    patterns: '/cognitive/patterns',
    inhibitory: '/cognitive/inhibitory',
    attention: '/cognitive/attention',
  },
  neural: {
    index: '/neural',
    activity: '/neural/activity',
    connectivity: '/neural/connectivity',
    spikes: '/neural/spikes',
  },
  knowledgeGraph: '/knowledge-graph',
  memory: {
    index: '/memory',
    performance: '/memory/performance',
    consolidation: '/memory/consolidation',
    usage: '/memory/usage',
  },
  tools: {
    index: '/tools',
    catalog: '/tools/catalog',
    testing: '/tools/testing',
    history: '/tools/history',
  },
  architecture: '/architecture',
};

// Helper to build deep links
export const buildDeepLink = (path: string, params?: Record<string, string>): string => {
  const url = new URL(path, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, value);
    });
  }
  return url.toString();
};