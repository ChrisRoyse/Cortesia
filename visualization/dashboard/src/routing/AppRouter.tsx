import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorBoundary } from '../components/common/ErrorBoundary';
import { DashboardLayout } from '../components/Layout/DashboardLayout';

// Lazy load main dashboard page
const DashboardPage = lazy(() => import('../pages/Dashboard'));

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