import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { selectAllTools } from '../stores/toolsSlice';

interface ToolCatalogLayoutProps {
  children: React.ReactNode;
  sidebar?: boolean;
  sidebarContent?: React.ReactNode;
  sidebarWidth?: number;
  className?: string;
}

interface LayoutState {
  sidebarCollapsed: boolean;
  sidebarWidth: number;
  isResizing: boolean;
}

export const ToolCatalogLayout: React.FC<ToolCatalogLayoutProps> = ({
  children,
  sidebar = true,
  sidebarContent,
  sidebarWidth = 280,
  className = ''
}) => {
  const [layoutState, setLayoutState] = useState<LayoutState>({
    sidebarCollapsed: false,
    sidebarWidth: sidebarWidth,
    isResizing: false
  });

  const tools = useSelector(selectAllTools);

  // Handle responsive design
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setLayoutState(prev => ({ ...prev, sidebarCollapsed: true }));
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Check initial size

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Handle sidebar resize
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setLayoutState(prev => ({ ...prev, isResizing: true }));

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = Math.min(Math.max(200, e.clientX), 400);
      setLayoutState(prev => ({ ...prev, sidebarWidth: newWidth }));
    };

    const handleMouseUp = () => {
      setLayoutState(prev => ({ ...prev, isResizing: false }));
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  // Toggle sidebar collapse
  const toggleSidebar = () => {
    setLayoutState(prev => ({ 
      ...prev, 
      sidebarCollapsed: !prev.sidebarCollapsed 
    }));
  };

  return (
    <div className={`tool-catalog-layout ${className}`}>
      {/* Sidebar */}
      {sidebar && (
        <>
          <div 
            className={`sidebar ${layoutState.sidebarCollapsed ? 'collapsed' : ''}`}
            style={{ 
              width: layoutState.sidebarCollapsed ? 0 : layoutState.sidebarWidth 
            }}
          >
            <div className="sidebar-header">
              <h3 className="sidebar-title">Tool Browser</h3>
              <button 
                onClick={toggleSidebar}
                className="sidebar-toggle-btn"
                aria-label={layoutState.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              >
                {layoutState.sidebarCollapsed ? '→' : '←'}
              </button>
            </div>

            <div className="sidebar-content-wrapper">
              {sidebarContent || (
                <DefaultSidebarContent tools={tools} />
              )}
            </div>
          </div>

          {/* Resize Handle */}
          {!layoutState.sidebarCollapsed && (
            <div 
              className="resize-handle"
              onMouseDown={handleMouseDown}
              style={{ cursor: layoutState.isResizing ? 'col-resize' : 'col-resize' }}
            />
          )}
        </>
      )}

      {/* Main Content */}
      <div className="main-content-area">
        {children}
      </div>

      {/* Overlay for mobile */}
      {sidebar && !layoutState.sidebarCollapsed && window.innerWidth < 768 && (
        <div 
          className="sidebar-overlay"
          onClick={toggleSidebar}
        />
      )}

      <style jsx>{`
        .tool-catalog-layout {
          display: flex;
          height: 100%;
          position: relative;
          background: var(--bg-primary);
          color: var(--text-primary);
        }

        .sidebar {
          display: flex;
          flex-direction: column;
          background: var(--bg-secondary);
          border-right: 1px solid var(--border-color);
          transition: width 0.3s ease;
          min-width: 0;
          overflow: hidden;
          z-index: 100;
        }

        .sidebar.collapsed {
          width: 0 !important;
          border-right: none;
        }

        .sidebar-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 1rem;
          border-bottom: 1px solid var(--border-color);
          background: var(--bg-tertiary);
          min-height: 3.5rem;
          box-sizing: border-box;
        }

        .sidebar-title {
          margin: 0;
          font-size: 0.9rem;
          font-weight: 600;
          color: var(--text-primary);
        }

        .sidebar-toggle-btn {
          background: var(--bg-primary);
          border: 1px solid var(--border-color);
          color: var(--text-secondary);
          width: 24px;
          height: 24px;
          border-radius: 4px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 0.75rem;
          transition: all 0.2s;
        }

        .sidebar-toggle-btn:hover {
          border-color: var(--accent-color);
          color: var(--text-primary);
        }

        .sidebar-content-wrapper {
          flex: 1;
          overflow-y: auto;
          overflow-x: hidden;
        }

        .resize-handle {
          width: 4px;
          background: transparent;
          cursor: col-resize;
          position: relative;
          transition: background 0.2s;
        }

        .resize-handle:hover {
          background: var(--accent-color);
        }

        .resize-handle:active {
          background: var(--accent-color);
        }

        .main-content-area {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          min-width: 0;
        }

        .sidebar-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          z-index: 99;
        }

        /* Mobile responsive */
        @media (max-width: 768px) {
          .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            z-index: 101;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
          }

          .resize-handle {
            display: none;
          }
        }

        /* Animation for resize handle */
        .resize-handle::after {
          content: '';
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 2px;
          height: 20px;
          background: var(--border-color);
          border-radius: 1px;
          transition: background 0.2s;
        }

        .resize-handle:hover::after {
          background: var(--accent-color);
        }

        /* Scrollbar styling for sidebar */
        .sidebar-content-wrapper::-webkit-scrollbar {
          width: 6px;
        }

        .sidebar-content-wrapper::-webkit-scrollbar-track {
          background: var(--bg-secondary);
        }

        .sidebar-content-wrapper::-webkit-scrollbar-thumb {
          background: var(--border-color);
          border-radius: 3px;
        }

        .sidebar-content-wrapper::-webkit-scrollbar-thumb:hover {
          background: var(--text-secondary);
        }
      `}</style>
    </div>
  );
};

// Default sidebar content when none provided
const DefaultSidebarContent: React.FC<{ tools: any[] }> = ({ tools }) => {
  const categories = [
    'knowledge-graph',
    'neural', 
    'cognitive',
    'memory',
    'analysis',
    'utility'
  ];

  const statusCounts = {
    healthy: tools.filter(t => t.status?.health === 'healthy').length,
    degraded: tools.filter(t => t.status?.health === 'degraded').length,
    unavailable: tools.filter(t => t.status?.health === 'unavailable').length,
    unknown: tools.filter(t => t.status?.health === 'unknown').length,
  };

  return (
    <div className="default-sidebar-content">
      {/* Tool Categories */}
      <div className="sidebar-section">
        <h4 className="section-title">Categories</h4>
        <div className="category-list">
          {categories.map(category => {
            const count = tools.filter(t => t.category === category).length;
            return (
              <div key={category} className="category-item">
                <span className="category-name">
                  {category.replace('-', ' ')}
                </span>
                <span className="category-count">{count}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Status Overview */}
      <div className="sidebar-section">
        <h4 className="section-title">Health Status</h4>
        <div className="status-list">
          {Object.entries(statusCounts).map(([status, count]) => (
            <div key={status} className={`status-item ${status}`}>
              <span className="status-indicator"></span>
              <span className="status-name">{status}</span>
              <span className="status-count">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="sidebar-section">
        <h4 className="section-title">Quick Stats</h4>
        <div className="stats-grid">
          <div className="stat-item">
            <div className="stat-value">{tools.length}</div>
            <div className="stat-label">Total Tools</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">
              {tools.reduce((sum, t) => sum + (t.metrics?.totalExecutions || 0), 0)}
            </div>
            <div className="stat-label">Total Runs</div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .default-sidebar-content {
          padding: 1rem;
        }

        .sidebar-section {
          margin-bottom: 1.5rem;
        }

        .section-title {
          margin: 0 0 0.75rem 0;
          font-size: 0.8rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: var(--text-secondary);
        }

        .category-list {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .category-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem 0.75rem;
          border-radius: 4px;
          background: var(--bg-primary);
          transition: background 0.2s;
          cursor: pointer;
        }

        .category-item:hover {
          background: var(--bg-hover);
        }

        .category-name {
          font-size: 0.85rem;
          color: var(--text-primary);
          text-transform: capitalize;
        }

        .category-count {
          font-size: 0.75rem;
          color: var(--text-secondary);
          background: var(--bg-secondary);
          padding: 0.15rem 0.4rem;
          border-radius: 10px;
          min-width: 1.2rem;
          text-align: center;
        }

        .status-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .status-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.85rem;
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

        .status-item.unknown .status-indicator {
          background: var(--text-secondary);
        }

        .status-name {
          flex: 1;
          text-transform: capitalize;
          color: var(--text-primary);
        }

        .status-count {
          font-size: 0.75rem;
          color: var(--text-secondary);
          background: var(--bg-primary);
          padding: 0.15rem 0.4rem;
          border-radius: 10px;
          min-width: 1.2rem;
          text-align: center;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.75rem;
        }

        .stat-item {
          background: var(--bg-primary);
          padding: 0.75rem;
          border-radius: 6px;
          text-align: center;
        }

        .stat-value {
          display: block;
          font-size: 1.25rem;
          font-weight: 600;
          color: var(--accent-color);
          line-height: 1.2;
        }

        .stat-label {
          display: block;
          font-size: 0.7rem;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-top: 0.25rem;
        }
      `}</style>
    </div>
  );
};