import React, { useState, useMemo } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import {
  CpuChipIcon,
  CircuitBoardIcon,
  ShareIcon,
  DatabaseIcon,
  WrenchScrewdriverIcon,
  CubeTransparentIcon,
  HomeIcon,
  MagnifyingGlassIcon,
  StarIcon,
} from '@heroicons/react/24/outline';
import { useRealTimeStatus } from '../../hooks/useRealTimeStatus';
import { StatusIndicator } from '../common/StatusIndicator';
import { SearchBox } from '../common/SearchBox';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

interface NavigationItem {
  id: string;
  title: string;
  path?: string;
  icon: React.ComponentType<any>;
  badge?: string;
  children?: NavigationItem[];
  description?: string;
  status?: 'active' | 'warning' | 'error' | 'disabled';
}

const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    title: 'Dashboard',
    path: '/',
    icon: HomeIcon,
    description: 'System overview and key metrics',
  },
  {
    id: 'cognitive',
    title: 'Cognitive Patterns',
    icon: CpuChipIcon,
    description: 'Pattern recognition and cognitive processes',
    children: [
      {
        id: 'cognitive-overview',
        title: 'Overview',
        path: '/cognitive',
        icon: CpuChipIcon,
        description: 'Cognitive system overview',
      },
      {
        id: 'patterns',
        title: 'Pattern Recognition',
        path: '/cognitive/patterns',
        icon: MagnifyingGlassIcon,
        description: 'Pattern detection and analysis',
      },
      {
        id: 'inhibitory',
        title: 'Inhibitory Mechanisms',
        path: '/cognitive/inhibitory',
        icon: CpuChipIcon,
        description: 'Inhibition and control systems',
      },
      {
        id: 'attention',
        title: 'Attention System',
        path: '/cognitive/attention',
        icon: CpuChipIcon,
        description: 'Attention and focus mechanisms',
      },
    ],
  },
  {
    id: 'neural',
    title: 'Neural Activity',
    icon: CircuitBoardIcon,
    description: 'Neural network activity and connectivity',
    children: [
      {
        id: 'neural-overview',
        title: 'Overview',
        path: '/neural',
        icon: CircuitBoardIcon,
        description: 'Neural system overview',
      },
      {
        id: 'activity',
        title: 'Activity Heatmaps',
        path: '/neural/activity',
        icon: CircuitBoardIcon,
        description: 'Real-time neural activity visualization',
      },
      {
        id: 'connectivity',
        title: 'Connectivity Maps',
        path: '/neural/connectivity',
        icon: ShareIcon,
        description: 'Neural connection patterns',
      },
      {
        id: 'spikes',
        title: 'Spike Analysis',
        path: '/neural/spikes',
        icon: CircuitBoardIcon,
        description: 'Spike timing and frequency analysis',
      },
    ],
  },
  {
    id: 'knowledge-graph',
    title: 'Knowledge Graph',
    path: '/knowledge-graph',
    icon: ShareIcon,
    description: '3D knowledge graph visualization',
  },
  {
    id: 'memory',
    title: 'Memory Systems',
    icon: DatabaseIcon,
    description: 'Memory performance and consolidation',
    children: [
      {
        id: 'memory-overview',
        title: 'Overview',
        path: '/memory',
        icon: DatabaseIcon,
        description: 'Memory system overview',
      },
      {
        id: 'performance',
        title: 'Performance Metrics',
        path: '/memory/performance',
        icon: DatabaseIcon,
        description: 'Memory usage and performance',
      },
      {
        id: 'consolidation',
        title: 'Consolidation Monitor',
        path: '/memory/consolidation',
        icon: DatabaseIcon,
        description: 'Memory consolidation processes',
      },
      {
        id: 'usage',
        title: 'Usage Analytics',
        path: '/memory/usage',
        icon: DatabaseIcon,
        description: 'Memory access patterns',
      },
    ],
  },
  {
    id: 'tools',
    title: 'MCP Tools',
    icon: WrenchScrewdriverIcon,
    description: 'Model Context Protocol tools',
    children: [
      {
        id: 'tools-overview',
        title: 'Overview',
        path: '/tools',
        icon: WrenchScrewdriverIcon,
        description: 'Tools system overview',
      },
      {
        id: 'catalog',
        title: 'Tool Catalog',
        path: '/tools/catalog',
        icon: WrenchScrewdriverIcon,
        description: 'Available MCP tools',
      },
      {
        id: 'testing',
        title: 'Tool Testing',
        path: '/tools/testing',
        icon: WrenchScrewdriverIcon,
        description: 'Interactive tool testing',
      },
      {
        id: 'history',
        title: 'Execution History',
        path: '/tools/history',
        icon: WrenchScrewdriverIcon,
        description: 'Tool execution logs',
      },
    ],
  },
  {
    id: 'architecture',
    title: 'System Architecture',
    path: '/architecture',
    icon: CubeTransparentIcon,
    description: 'System components and health',
  },
];

export const Sidebar: React.FC<SidebarProps> = ({ isOpen, onToggle }) => {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const location = useLocation();
  const { systemStatus, componentStatuses } = useRealTimeStatus();

  const toggleExpanded = (itemId: string) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(itemId)) {
      newExpanded.delete(itemId);
    } else {
      newExpanded.add(itemId);
    }
    setExpandedItems(newExpanded);
  };

  const toggleFavorite = (itemId: string) => {
    const newFavorites = new Set(favorites);
    if (newFavorites.has(itemId)) {
      newFavorites.delete(itemId);
    } else {
      newFavorites.add(itemId);
    }
    setFavorites(newFavorites);
  };

  const getItemStatus = (itemId: string) => {
    switch (itemId) {
      case 'cognitive':
        return componentStatuses.cognitive?.status || 'active';
      case 'neural':
        return componentStatuses.neural?.status || 'active';
      case 'memory':
        return componentStatuses.memory?.status || 'active';
      case 'tools':
        return componentStatuses.tools?.status || 'active';
      default:
        return 'active';
    }
  };

  const filteredItems = useMemo(() => {
    if (!searchQuery) return navigationItems;
    
    const filterItem = (item: NavigationItem): NavigationItem | null => {
      const matchesQuery = item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          item.description?.toLowerCase().includes(searchQuery.toLowerCase());
      
      if (item.children) {
        const filteredChildren = item.children
          .map(child => filterItem(child))
          .filter(Boolean) as NavigationItem[];
        
        if (matchesQuery || filteredChildren.length > 0) {
          return { ...item, children: filteredChildren };
        }
      } else if (matchesQuery) {
        return item;
      }
      
      return null;
    };

    return navigationItems
      .map(item => filterItem(item))
      .filter(Boolean) as NavigationItem[];
  }, [searchQuery]);

  const renderNavigationItem = (item: NavigationItem, level = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.has(item.id);
    const isFavorite = favorites.has(item.id);
    const isActive = item.path ? location.pathname === item.path : false;
    const status = getItemStatus(item.id);

    const itemClass = `
      flex items-center w-full px-3 py-2 text-left transition-colors duration-200 rounded-lg group
      ${level === 0 ? 'text-gray-700 hover:bg-blue-50 hover:text-blue-700' : 'text-gray-600 hover:bg-gray-50'}
      ${isActive ? 'bg-blue-100 text-blue-700 font-medium' : ''}
      ${level > 0 ? 'ml-4 text-sm' : ''}
    `;

    const content = (
      <>
        <div className="flex items-center flex-1 min-w-0">
          <item.icon className={`
            w-5 h-5 flex-shrink-0 mr-3
            ${isActive ? 'text-blue-600' : 'text-gray-400 group-hover:text-gray-600'}
          `} />
          <span className="truncate">{item.title}</span>
          {status !== 'active' && (
            <StatusIndicator 
              status={status} 
              size="sm" 
              className="ml-2 flex-shrink-0"
            />
          )}
        </div>
        
        <div className="flex items-center space-x-1 flex-shrink-0">
          {isFavorite && (
            <StarIcon className="w-4 h-4 text-yellow-400" />
          )}
          {hasChildren && (
            <button
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                toggleExpanded(item.id);
              }}
              className="p-1 rounded-md hover:bg-gray-200"
            >
              {isExpanded ? (
                <ChevronDownIcon className="w-4 h-4" />
              ) : (
                <ChevronRightIcon className="w-4 h-4" />
              )}
            </button>
          )}
        </div>
      </>
    );

    return (
      <div key={item.id}>
        {item.path ? (
          <NavLink
            to={item.path}
            className={itemClass}
            title={item.description}
            onClick={() => {
              if (!isOpen && window.innerWidth < 768) {
                onToggle();
              }
            }}
          >
            {content}
          </NavLink>
        ) : (
          <button
            className={itemClass}
            onClick={() => toggleExpanded(item.id)}
            title={item.description}
          >
            {content}
          </button>
        )}
        
        {hasChildren && isExpanded && (
          <div className="mt-1 space-y-1">
            {item.children?.map(child => renderNavigationItem(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-20 bg-black bg-opacity-50 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-30 w-80 bg-white border-r border-gray-200 transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <h1 className="text-xl font-bold text-gray-900">LLMKG Dashboard</h1>
            <button
              onClick={onToggle}
              className="p-2 rounded-md hover:bg-gray-100 lg:hidden"
            >
              <span className="sr-only">Close sidebar</span>
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Search */}
          <div className="p-4 border-b border-gray-200">
            <SearchBox
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder="Search features..."
              className="w-full"
            />
          </div>

          {/* System Status */}
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">System Status</span>
              <StatusIndicator status={systemStatus.overall} />
            </div>
            <div className="mt-2 text-xs text-gray-500">
              Last updated: {new Date().toLocaleTimeString()}
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 overflow-y-auto">
            <div className="space-y-2">
              {filteredItems.map(item => renderNavigationItem(item))}
            </div>

            {/* Favorites Section */}
            {favorites.size > 0 && (
              <div className="mt-8">
                <h3 className="px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  Favorites
                </h3>
                <div className="mt-2 space-y-1">
                  {Array.from(favorites).map(favoriteId => {
                    const item = navigationItems.find(item => item.id === favoriteId);
                    return item ? renderNavigationItem(item) : null;
                  })}
                </div>
              </div>
            )}
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-gray-200">
            <div className="text-xs text-gray-500 text-center">
              LLMKG v1.0.0
              <br />
              Brain-inspired Knowledge Graph
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;