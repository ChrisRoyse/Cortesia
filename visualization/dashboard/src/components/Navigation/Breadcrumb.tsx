import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ChevronRightIcon, HomeIcon } from '@heroicons/react/24/outline';

interface BreadcrumbItem {
  label: string;
  path: string;
  icon?: React.ComponentType<any>;
  isActive?: boolean;
}

interface BreadcrumbProps {
  className?: string;
  showHome?: boolean;
  maxItems?: number;
}

// Route metadata for generating breadcrumbs
const routeMetadata: Record<string, { label: string; icon?: React.ComponentType<any> }> = {
  '/': { label: 'Dashboard', icon: HomeIcon },
  '/cognitive': { label: 'Cognitive Patterns' },
  '/cognitive/patterns': { label: 'Pattern Recognition' },
  '/cognitive/inhibitory': { label: 'Inhibitory Mechanisms' },
  '/cognitive/attention': { label: 'Attention System' },
  '/neural': { label: 'Neural Activity' },
  '/neural/activity': { label: 'Activity Heatmaps' },
  '/neural/connectivity': { label: 'Connectivity Maps' },
  '/neural/spikes': { label: 'Spike Analysis' },
  '/knowledge-graph': { label: 'Knowledge Graph' },
  '/memory': { label: 'Memory Systems' },
  '/memory/performance': { label: 'Performance Metrics' },
  '/memory/consolidation': { label: 'Consolidation Monitor' },
  '/memory/usage': { label: 'Usage Analytics' },
  '/tools': { label: 'MCP Tools' },
  '/tools/catalog': { label: 'Tool Catalog' },
  '/tools/testing': { label: 'Tool Testing' },
  '/tools/history': { label: 'Execution History' },
  '/architecture': { label: 'System Architecture' },
};

export const Breadcrumb: React.FC<BreadcrumbProps> = ({
  className = '',
  showHome = true,
  maxItems = 5,
}) => {
  const location = useLocation();
  
  const generateBreadcrumbs = (): BreadcrumbItem[] => {
    const pathSegments = location.pathname.split('/').filter(Boolean);
    const breadcrumbs: BreadcrumbItem[] = [];

    // Add home if requested
    if (showHome && location.pathname !== '/') {
      breadcrumbs.push({
        label: routeMetadata['/'].label,
        path: '/',
        icon: routeMetadata['/'].icon,
      });
    }

    // Build breadcrumbs from path segments
    let currentPath = '';
    pathSegments.forEach((segment, index) => {
      currentPath += `/${segment}`;
      const metadata = routeMetadata[currentPath];
      
      if (metadata) {
        breadcrumbs.push({
          label: metadata.label,
          path: currentPath,
          icon: metadata.icon,
          isActive: index === pathSegments.length - 1,
        });
      }
    });

    // Truncate if too many items
    if (breadcrumbs.length > maxItems) {
      const truncatedBreadcrumbs = [
        breadcrumbs[0], // Always show first item
        { label: '...', path: '#', isActive: false },
        ...breadcrumbs.slice(-2), // Show last two items
      ];
      return truncatedBreadcrumbs;
    }

    return breadcrumbs;
  };

  const breadcrumbs = generateBreadcrumbs();

  if (breadcrumbs.length <= 1) {
    return null; // Don't show breadcrumbs for single-level pages
  }

  return (
    <nav
      className={`flex items-center space-x-1 text-sm ${className}`}
      aria-label="Breadcrumb"
    >
      <ol className="flex items-center space-x-1">
        {breadcrumbs.map((item, index) => {
          const isLast = index === breadcrumbs.length - 1;
          const isEllipsis = item.label === '...';

          return (
            <li key={item.path} className="flex items-center">
              {index > 0 && (
                <ChevronRightIcon className="w-4 h-4 text-gray-400 mx-2" />
              )}
              
              {isEllipsis ? (
                <span className="text-gray-400 px-2">...</span>
              ) : isLast ? (
                <span
                  className="flex items-center text-gray-900 font-medium"
                  aria-current="page"
                >
                  {item.icon && (
                    <item.icon className="w-4 h-4 mr-2 text-gray-600" />
                  )}
                  {item.label}
                </span>
              ) : (
                <Link
                  to={item.path}
                  className="flex items-center text-gray-500 hover:text-gray-700 transition-colors duration-200"
                >
                  {item.icon && (
                    <item.icon className="w-4 h-4 mr-2" />
                  )}
                  {item.label}
                </Link>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
};

// Enhanced breadcrumb with additional features
interface EnhancedBreadcrumbProps extends BreadcrumbProps {
  customItems?: BreadcrumbItem[];
  showIcons?: boolean;
  showHome?: boolean;
  separator?: React.ComponentType<any>;
}

export const EnhancedBreadcrumb: React.FC<EnhancedBreadcrumbProps> = ({
  customItems,
  showIcons = true,
  showHome = true,
  separator = ChevronRightIcon,
  className = '',
  maxItems = 5,
}) => {
  const location = useLocation();
  
  const getBreadcrumbs = (): BreadcrumbItem[] => {
    if (customItems) {
      return customItems;
    }

    const pathSegments = location.pathname.split('/').filter(Boolean);
    const breadcrumbs: BreadcrumbItem[] = [];

    if (showHome && location.pathname !== '/') {
      breadcrumbs.push({
        label: 'Dashboard',
        path: '/',
        icon: showIcons ? HomeIcon : undefined,
      });
    }

    let currentPath = '';
    pathSegments.forEach((segment, index) => {
      currentPath += `/${segment}`;
      const metadata = routeMetadata[currentPath];
      
      if (metadata) {
        breadcrumbs.push({
          label: metadata.label,
          path: currentPath,
          icon: showIcons ? metadata.icon : undefined,
          isActive: index === pathSegments.length - 1,
        });
      }
    });

    return breadcrumbs;
  };

  const breadcrumbs = getBreadcrumbs();
  const Separator = separator;

  if (breadcrumbs.length <= 1) {
    return null;
  }

  return (
    <nav
      className={`flex items-center ${className}`}
      aria-label="Enhanced breadcrumb navigation"
    >
      <div className="flex items-center space-x-1 bg-white rounded-lg px-3 py-2 shadow-sm border border-gray-200">
        {breadcrumbs.map((item, index) => {
          const isLast = index === breadcrumbs.length - 1;

          return (
            <React.Fragment key={item.path}>
              {index > 0 && (
                <Separator className="w-4 h-4 text-gray-400 mx-1" />
              )}
              
              {isLast ? (
                <span
                  className="flex items-center text-gray-900 font-medium text-sm"
                  aria-current="page"
                >
                  {item.icon && showIcons && (
                    <item.icon className="w-4 h-4 mr-2 text-gray-600" />
                  )}
                  {item.label}
                </span>
              ) : (
                <Link
                  to={item.path}
                  className="flex items-center text-gray-500 hover:text-blue-600 transition-colors duration-200 text-sm px-2 py-1 rounded hover:bg-gray-50"
                >
                  {item.icon && showIcons && (
                    <item.icon className="w-4 h-4 mr-2" />
                  )}
                  {item.label}
                </Link>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </nav>
  );
};

// Context menu breadcrumb for additional actions
interface ContextBreadcrumbProps extends BreadcrumbProps {
  actions?: Array<{
    label: string;
    icon?: React.ComponentType<any>;
    onClick: () => void;
  }>;
}

export const ContextBreadcrumb: React.FC<ContextBreadcrumbProps> = ({
  actions = [],
  className = '',
  ...breadcrumbProps
}) => {
  return (
    <div className={`flex items-center justify-between ${className}`}>
      <Breadcrumb {...breadcrumbProps} />
      
      {actions.length > 0 && (
        <div className="flex items-center space-x-2 ml-4">
          {actions.map((action, index) => (
            <button
              key={index}
              onClick={action.onClick}
              className="flex items-center px-3 py-1 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors duration-200"
            >
              {action.icon && <action.icon className="w-4 h-4 mr-1" />}
              {action.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default Breadcrumb;