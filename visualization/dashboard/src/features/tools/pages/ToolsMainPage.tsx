import React, { useState, useEffect, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { 
  selectAllTools, 
  selectFilteredTools, 
  selectToolsLoading, 
  selectToolsError,
  selectActiveExecutions,
  selectExecutionHistory,
  setView,
  selectTool,
  setSearchTerm,
  clearFilters 
} from '../stores/toolsSlice';
import { useToolDiscovery } from '../hooks/useToolDiscovery';
import { useToolStatus } from '../hooks/useToolStatus';
import { useToolAnalytics } from '../hooks/useToolAnalytics';

// Component imports
import { ToolCatalogLayout } from '../components/ToolCatalogLayout';
import { ToolCatalog } from '../components/catalog/ToolCatalog';
import { ToolDetailsDialog } from '../components/catalog/ToolDetailsDialog';
import { StatusDashboard } from '../components/monitoring/StatusDashboard';
import { ToolTester } from '../components/testing/ToolTester';
import { PerformanceDashboard } from '../components/analytics/PerformanceDashboard';
import { ToolDocViewer } from '../components/documentation/ToolDocViewer';
import { ExecutionHistory } from '../components/testing/ExecutionHistory';

// Navigation tabs configuration
const NAVIGATION_TABS = [
  { id: 'catalog', label: 'Tool Catalog', icon: '🔧', path: '/tools' },
  { id: 'monitor', label: 'Status Monitor', icon: '📊', path: '/tools/monitor' },
  { id: 'test', label: 'Tool Tester', icon: '🧪', path: '/tools/test' },
  { id: 'analytics', label: 'Analytics', icon: '📈', path: '/tools/analytics' },
  { id: 'docs', label: 'Documentation', icon: '📚', path: '/tools/docs' },
  { id: 'history', label: 'Execution History', icon: '📋', path: '/tools/history' }
] as const;

interface ToolsMainPageProps {
  className?: string;
}

export const ToolsMainPage: React.FC<ToolsMainPageProps> = ({ className }) => {
  const dispatch = useDispatch();
  const location = useLocation();
  const navigate = useNavigate();

  // Redux state
  const tools = useSelector(selectAllTools);
  const filteredTools = useSelector(selectFilteredTools);
  const loading = useSelector(selectToolsLoading);
  const error = useSelector(selectToolsError);
  const activeExecutions = useSelector(selectActiveExecutions);
  const executionHistory = useSelector(selectExecutionHistory);

  // Local state
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [showToolDetails, setShowToolDetails] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [quickSearch, setQuickSearch] = useState('');

  // Custom hooks for real-time features
  const { 
    discoveredTools, 
    isDiscovering, 
    discoverTools, 
    registerNewTools 
  } = useToolDiscovery();

  const { 
    statusUpdates, 
    monitoringActive, 
    startMonitoring, 
    stopMonitoring 
  } = useToolStatus();

  const { 
    analyticsData, 
    refreshAnalytics, 
    exportAnalytics 
  } = useToolAnalytics();

  // Initialize services on mount
  useEffect(() => {
    const initializeServices = async () => {
      try {
        // Start tool discovery
        await discoverTools();
        
        // Start status monitoring
        startMonitoring();
        
        // Refresh analytics
        await refreshAnalytics();
      } catch (err) {
        console.error('Failed to initialize tool services:', err);
      }
    };

    initializeServices();

    return () => {
      // Cleanup on unmount
      stopMonitoring();
    };
  }, []);

  // Handle real-time tool registration
  useEffect(() => {
    if (discoveredTools.length > 0) {
      registerNewTools(discoveredTools);
    }
  }, [discoveredTools, registerNewTools]);

  // Handle quick search
  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      dispatch(setSearchTerm(quickSearch));
    }, 300);

    return () => clearTimeout(debounceTimer);
  }, [quickSearch, dispatch]);

  // Get current active tab based on route
  const currentTab = useMemo(() => {
    const currentPath = location.pathname;
    return NAVIGATION_TABS.find(tab => 
      tab.path === currentPath || currentPath.startsWith(tab.path + '/')
    ) || NAVIGATION_TABS[0];
  }, [location.pathname]);

  // Handle tool selection
  const handleToolSelect = (toolId: string) => {
    setSelectedToolId(toolId);
    dispatch(selectTool(toolId));
    setShowToolDetails(true);
  };

  // Handle navigation
  const handleNavigate = (tabId: string) => {
    const tab = NAVIGATION_TABS.find(t => t.id === tabId);
    if (tab) {
      navigate(tab.path);
    }
  };

  // Handle view changes
  const handleViewChange = (view: 'grid' | 'list' | 'table') => {
    dispatch(setView(view));
  };

  // Handle filter clear
  const handleClearFilters = () => {
    dispatch(clearFilters());
    setQuickSearch('');
  };

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    const healthyTools = tools.filter(t => t.status.health === 'healthy').length;
    const degradedTools = tools.filter(t => t.status.health === 'degraded').length;
    const unavailableTools = tools.filter(t => t.status.health === 'unavailable').length;
    const totalExecutions = tools.reduce((sum, t) => sum + t.metrics.totalExecutions, 0);
    const averageSuccessRate = tools.length > 0 
      ? tools.reduce((sum, t) => sum + t.metrics.successRate, 0) / tools.length 
      : 0;

    return {
      totalTools: tools.length,
      healthyTools,
      degradedTools,
      unavailableTools,
      totalExecutions,
      averageSuccessRate,
      activeExecutions: activeExecutions.length,
    };
  }, [tools, activeExecutions]);

  return (
    <div className={`tools-main-page ${className || ''}`}>
      {/* Page Header */}
      <div className="page-header">
        <div className="header-content">
          <div className="header-info">
            <h1 className="page-title">
              MCP Tool Catalog
              {isDiscovering && <div className="discovering-indicator" />}
            </h1>
            <p className="page-description">
              Comprehensive tool management, monitoring, and analytics for LLMKG system
            </p>
          </div>
          
          {/* Quick Stats */}
          <div className="quick-stats">
            <div className="stat-item">
              <span className="stat-value">{summaryStats.totalTools}</span>
              <span className="stat-label">Total Tools</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{summaryStats.healthyTools}</span>
              <span className="stat-label">Healthy</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{summaryStats.activeExecutions}</span>
              <span className="stat-label">Active</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">
                {(summaryStats.averageSuccessRate * 100).toFixed(1)}%
              </span>
              <span className="stat-label">Success Rate</span>
            </div>
          </div>
        </div>

        {/* Quick Search */}
        <div className="header-actions">
          <div className="quick-search">
            <input
              type="text"
              placeholder="Quick search tools..."
              value={quickSearch}
              onChange={(e) => setQuickSearch(e.target.value)}
              className="search-input"
            />
            {quickSearch && (
              <button 
                onClick={() => setQuickSearch('')}
                className="clear-search"
              >
                ✕
              </button>
            )}
          </div>
          
          <button 
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="sidebar-toggle"
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? '→' : '←'}
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="navigation-tabs">
        {NAVIGATION_TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => handleNavigate(tab.id)}
            className={`nav-tab ${currentTab.id === tab.id ? 'active' : ''}`}
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        <ToolCatalogLayout
          sidebar={!sidebarCollapsed}
          sidebarContent={
            <div className="sidebar-content">
              {/* Tool Categories */}
              <div className="sidebar-section">
                <h3>Categories</h3>
                <div className="category-list">
                  {['knowledge-graph', 'neural', 'cognitive', 'memory', 'analysis', 'utility'].map(category => {
                    const count = tools.filter(t => t.category === category).length;
                    return (
                      <div key={category} className="category-item">
                        <span className="category-name">{category}</span>
                        <span className="category-count">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Status Summary */}
              <div className="sidebar-section">
                <h3>Status Overview</h3>
                <div className="status-summary">
                  <div className="status-item healthy">
                    <span className="status-indicator"></span>
                    <span>Healthy: {summaryStats.healthyTools}</span>
                  </div>
                  <div className="status-item degraded">
                    <span className="status-indicator"></span>
                    <span>Degraded: {summaryStats.degradedTools}</span>
                  </div>
                  <div className="status-item unavailable">
                    <span className="status-indicator"></span>
                    <span>Unavailable: {summaryStats.unavailableTools}</span>
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              <div className="sidebar-section">
                <h3>Recent Activity</h3>
                <div className="recent-activity">
                  {executionHistory.slice(0, 5).map((execution, index) => (
                    <div key={execution.id} className="activity-item">
                      <div className="activity-tool">{execution.toolName}</div>
                      <div className="activity-time">
                        {new Date(execution.startTime).toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          }
        >
          {/* Route-based content */}
          <div className="route-content">
            {error && (
              <div className="error-banner">
                <span className="error-icon">⚠️</span>
                <span className="error-message">{error}</span>
                <button onClick={handleClearFilters} className="error-action">
                  Clear Filters
                </button>
              </div>
            )}

            <Routes>
              <Route 
                path="/" 
                element={
                  <ToolCatalog 
                    tools={filteredTools}
                    loading={loading}
                    onToolSelect={handleToolSelect}
                    onViewChange={handleViewChange}
                  />
                } 
              />
              <Route 
                path="/monitor" 
                element={
                  <StatusDashboard 
                    tools={tools}
                    statusUpdates={statusUpdates}
                    monitoringActive={monitoringActive}
                  />
                } 
              />
              <Route 
                path="/test/*" 
                element={
                  <ToolTester 
                    toolId={selectedToolId}
                    tools={tools}
                    onToolSelect={handleToolSelect}
                  />
                } 
              />
              <Route 
                path="/analytics" 
                element={
                  <PerformanceDashboard 
                    analyticsData={analyticsData}
                    tools={tools}
                    onExport={exportAnalytics}
                    onRefresh={refreshAnalytics}
                  />
                } 
              />
              <Route 
                path="/docs/*" 
                element={
                  <ToolDocViewer 
                    toolId={selectedToolId}
                    tools={tools}
                    onToolSelect={handleToolSelect}
                  />
                } 
              />
              <Route 
                path="/history" 
                element={
                  <ExecutionHistory 
                    history={executionHistory}
                    activeExecutions={activeExecutions}
                  />
                } 
              />
              {/* Redirect unknown routes to catalog */}
              <Route path="*" element={<Navigate to="/tools" replace />} />
            </Routes>
          </div>
        </ToolCatalogLayout>
      </div>

      {/* Tool Details Modal */}
      {showToolDetails && selectedToolId && (
        <ToolDetailsDialog
          toolId={selectedToolId}
          isOpen={showToolDetails}
          onClose={() => {
            setShowToolDetails(false);
            setSelectedToolId(null);
            dispatch(selectTool(null));
          }}
        />
      )}

      {/* Loading Overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner"></div>
            <div className="loading-text">Loading tools...</div>
          </div>
        </div>
      )}

      <style jsx>{`
        .tools-main-page {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: var(--bg-primary);
          color: var(--text-primary);
        }

        .page-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1.5rem 2rem;
          border-bottom: 1px solid var(--border-color);
          background: var(--bg-secondary);
        }

        .header-content {
          display: flex;
          align-items: center;
          gap: 2rem;
        }

        .header-info {
          flex: 1;
        }

        .page-title {
          margin: 0 0 0.5rem 0;
          font-size: 1.75rem;
          font-weight: 600;
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .discovering-indicator {
          width: 8px;
          height: 8px;
          background: var(--accent-color);
          border-radius: 50%;
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }

        .page-description {
          margin: 0;
          color: var(--text-secondary);
          font-size: 0.9rem;
        }

        .quick-stats {
          display: flex;
          gap: 1.5rem;
        }

        .stat-item {
          text-align: center;
        }

        .stat-value {
          display: block;
          font-size: 1.5rem;
          font-weight: 600;
          color: var(--accent-color);
          line-height: 1.2;
        }

        .stat-label {
          display: block;
          font-size: 0.75rem;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .header-actions {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .quick-search {
          position: relative;
          display: flex;
          align-items: center;
        }

        .search-input {
          width: 250px;
          padding: 0.5rem 0.75rem;
          background: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: 6px;
          color: var(--text-primary);
          font-size: 0.875rem;
          transition: border-color 0.2s, box-shadow 0.2s;
        }

        .search-input:focus {
          outline: none;
          border-color: var(--accent-color);
          box-shadow: 0 0 0 3px rgba(0, 122, 204, 0.1);
        }

        .clear-search {
          position: absolute;
          right: 0.5rem;
          background: none;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
          padding: 0.25rem;
          line-height: 1;
          transition: color 0.2s;
        }

        .clear-search:hover {
          color: var(--text-primary);
        }

        .sidebar-toggle {
          background: var(--bg-primary);
          border: 1px solid var(--border-color);
          color: var(--text-secondary);
          padding: 0.5rem;
          border-radius: 4px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .sidebar-toggle:hover {
          border-color: var(--accent-color);
          color: var(--text-primary);
        }

        .navigation-tabs {
          display: flex;
          background: var(--bg-secondary);
          border-bottom: 1px solid var(--border-color);
          overflow-x: auto;
        }

        .nav-tab {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 1rem;
          background: none;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
          border-bottom: 2px solid transparent;
        }

        .nav-tab:hover {
          color: var(--text-primary);
          background: var(--bg-hover);
        }

        .nav-tab.active {
          color: var(--accent-color);
          border-bottom-color: var(--accent-color);
        }

        .tab-icon {
          font-size: 1rem;
        }

        .tab-label {
          font-size: 0.875rem;
          font-weight: 500;
        }

        .main-content {
          flex: 1;
          display: flex;
          overflow: hidden;
        }

        .sidebar-content {
          padding: 1rem;
          width: 100%;
        }

        .sidebar-section {
          margin-bottom: 2rem;
        }

        .sidebar-section h3 {
          margin: 0 0 0.75rem 0;
          font-size: 0.875rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: var(--text-secondary);
        }

        .category-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .category-item {
          display: flex;
          justify-content: space-between;
          padding: 0.5rem 0.75rem;
          background: var(--bg-primary);
          border-radius: 4px;
          font-size: 0.875rem;
        }

        .category-name {
          text-transform: capitalize;
          color: var(--text-primary);
        }

        .category-count {
          color: var(--text-secondary);
          background: var(--bg-secondary);
          padding: 0.125rem 0.5rem;
          border-radius: 12px;
          font-size: 0.75rem;
          font-weight: 500;
          min-width: 1.5rem;
          text-align: center;
        }

        .status-summary {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .status-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.875rem;
        }

        .status-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .status-item.healthy .status-indicator {
          background: var(--success-color);
        }

        .status-item.degraded .status-indicator {
          background: var(--warning-color);
        }

        .status-item.unavailable .status-indicator {
          background: var(--error-color);
        }

        .recent-activity {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .activity-item {
          display: flex;
          justify-content: space-between;
          padding: 0.5rem 0;
          border-bottom: 1px solid var(--border-color);
          font-size: 0.875rem;
        }

        .activity-item:last-child {
          border-bottom: none;
        }

        .activity-tool {
          font-weight: 500;
          color: var(--text-primary);
          flex: 1;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .activity-time {
          color: var(--text-secondary);
          font-size: 0.75rem;
        }

        .route-content {
          flex: 1;
          overflow: hidden;
        }

        .error-banner {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 1rem;
          margin-bottom: 1rem;
          background: rgba(var(--error-color-rgb), 0.1);
          border: 1px solid var(--error-color);
          border-radius: 6px;
          color: var(--error-color);
        }

        .error-icon {
          font-size: 1.25rem;
        }

        .error-message {
          flex: 1;
          font-weight: 500;
        }

        .error-action {
          background: var(--error-color);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.875rem;
          font-weight: 500;
          transition: opacity 0.2s;
        }

        .error-action:hover {
          opacity: 0.9;
        }

        .loading-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .loading-spinner {
          background: var(--bg-secondary);
          padding: 2rem;
          border-radius: 8px;
          text-align: center;
        }

        .spinner {
          width: 32px;
          height: 32px;
          border: 3px solid var(--border-color);
          border-top-color: var(--accent-color);
          border-radius: 50%;
          margin: 0 auto 1rem;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .loading-text {
          color: var(--text-secondary);
          font-size: 0.875rem;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
          .page-header {
            flex-direction: column;
            gap: 1rem;
            padding: 1rem;
          }

          .header-content {
            width: 100%;
          }

          .quick-stats {
            gap: 1rem;
          }

          .search-input {
            width: 200px;
          }

          .navigation-tabs {
            padding: 0 0.5rem;
          }

          .nav-tab {
            padding: 0.5rem 0.75rem;
          }

          .tab-label {
            display: none;
          }
        }

        @media (max-width: 480px) {
          .quick-stats {
            display: none;
          }

          .search-input {
            width: 150px;
          }
        }
      `}</style>
    </div>
  );
};

export default ToolsMainPage;