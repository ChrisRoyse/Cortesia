import React, { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  WrenchScrewdriverIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  PlayIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon,
  BookOpenIcon,
  CodeBracketIcon,
  CommandLineIcon,
} from '@heroicons/react/24/outline';
import { useMCPTools } from '../../hooks/useMCPTools';
import { SearchBox } from '../../components/common/SearchBox';
import { StatusIndicator } from '../../components/common/StatusIndicator';
import { Badge } from '../../components/common/Badge';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { MetricCard } from '../../components/common/MetricCard';

interface ToolCategory {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  description: string;
  count: number;
}

const toolCategories: ToolCategory[] = [
  {
    id: 'filesystem',
    name: 'File System',
    icon: BookOpenIcon,
    description: 'File operations and directory management',
    count: 0,
  },
  {
    id: 'database',
    name: 'Database',
    icon: CommandLineIcon,
    description: 'Database queries and data manipulation',
    count: 0,
  },
  {
    id: 'api',
    name: 'API Tools',
    icon: CodeBracketIcon,
    description: 'HTTP requests and API interactions',
    count: 0,
  },
  {
    id: 'analysis',
    name: 'Analysis',
    icon: MagnifyingGlassIcon,
    description: 'Data analysis and processing tools',
    count: 0,
  },
  {
    id: 'utility',
    name: 'Utilities',
    icon: WrenchScrewdriverIcon,
    description: 'General purpose utility tools',
    count: 0,
  },
];

const ToolsPage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'usage' | 'recent'>('name');

  const {
    tools,
    recentExecutions,
    toolStats,
    isLoading,
    error,
    executeTools,
    refreshTools,
  } = useMCPTools();

  // Filter and sort tools
  const filteredTools = useMemo(() => {
    let filtered = tools;

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter(tool =>
        tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tool.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tool.category?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply category filter
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(tool => tool.category === selectedCategory);
    }

    // Sort tools
    switch (sortBy) {
      case 'usage':
        return filtered.sort((a, b) => (b.usage || 0) - (a.usage || 0));
      case 'recent':
        return filtered.sort((a, b) => 
          new Date(b.lastUsed || 0).getTime() - new Date(a.lastUsed || 0).getTime()
        );
      default:
        return filtered.sort((a, b) => a.name.localeCompare(b.name));
    }
  }, [tools, searchQuery, selectedCategory, sortBy]);

  // Update category counts
  const categoriesWithCounts = toolCategories.map(category => ({
    ...category,
    count: tools.filter(tool => tool.category === category.id).length,
  }));

  const getToolStatusIcon = (status: string) => {
    switch (status) {
      case 'available':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'error':
        return <XCircleIcon className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />;
      default:
        return <ClockIcon className="w-5 h-5 text-gray-400" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full min-h-[400px]">
        <LoadingSpinner size="large" message="Loading MCP tools..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <div className="flex items-center">
            <XCircleIcon className="w-6 h-6 text-red-500 mr-3" />
            <div>
              <h3 className="text-lg font-medium text-red-900">Error Loading Tools</h3>
              <p className="text-red-700 mt-1">{error}</p>
            </div>
          </div>
          <button
            onClick={refreshTools}
            className="mt-4 bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Page Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">MCP Tools</h1>
            <p className="mt-2 text-gray-600">
              Model Context Protocol tools for enhanced AI capabilities
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={refreshTools}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Refresh Tools
            </button>
          </div>
        </div>
      </div>

      {/* Tool Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Total Tools"
          value={tools.length.toString()}
          icon={WrenchScrewdriverIcon}
          className="bg-gradient-to-r from-blue-500 to-blue-600"
        />
        <MetricCard
          title="Executions Today"
          value={toolStats.executionsToday.toString()}
          change={`+${toolStats.executionGrowth}%`}
          changeType="increase"
          icon={PlayIcon}
          className="bg-gradient-to-r from-green-500 to-green-600"
        />
        <MetricCard
          title="Success Rate"
          value={`${toolStats.successRate}%`}
          icon={CheckCircleIcon}
          className="bg-gradient-to-r from-purple-500 to-purple-600"
        />
        <MetricCard
          title="Avg Response Time"
          value={`${toolStats.avgResponseTime}ms`}
          icon={ClockIcon}
          className="bg-gradient-to-r from-orange-500 to-orange-600"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          {/* Search and Filters */}
          <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Search & Filter
            </h3>
            
            <div className="space-y-4">
              <SearchBox
                value={searchQuery}
                onChange={setSearchQuery}
                placeholder="Search tools..."
                className="w-full"
              />

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Sort by
                </label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="name">Name</option>
                  <option value="usage">Usage</option>
                  <option value="recent">Recently Used</option>
                </select>
              </div>
            </div>
          </div>

          {/* Categories */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Categories
            </h3>
            <div className="space-y-2">
              <button
                onClick={() => setSelectedCategory('all')}
                className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                  selectedCategory === 'all'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span>All Tools</span>
                  <Badge variant="secondary">{tools.length}</Badge>
                </div>
              </button>
              
              {categoriesWithCounts.map((category) => (
                <button
                  key={category.id}
                  onClick={() => setSelectedCategory(category.id)}
                  className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                    selectedCategory === category.id
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <category.icon className="w-4 h-4 mr-2" />
                      <span>{category.name}</span>
                    </div>
                    <Badge variant="secondary">{category.count}</Badge>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3">
          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <Link
              to="/tools/catalog"
              className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow border border-gray-200 hover:border-blue-300"
            >
              <div className="flex items-center">
                <div className="p-3 bg-blue-100 rounded-lg">
                  <BookOpenIcon className="w-6 h-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Tool Catalog
                  </h3>
                  <p className="text-sm text-gray-600">Browse all available tools</p>
                </div>
              </div>
            </Link>

            <Link
              to="/tools/testing"
              className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow border border-gray-200 hover:border-green-300"
            >
              <div className="flex items-center">
                <div className="p-3 bg-green-100 rounded-lg">
                  <PlayIcon className="w-6 h-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Tool Testing
                  </h3>
                  <p className="text-sm text-gray-600">Test and debug tools</p>
                </div>
              </div>
            </Link>

            <Link
              to="/tools/history"
              className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow border border-gray-200 hover:border-purple-300"
            >
              <div className="flex items-center">
                <div className="p-3 bg-purple-100 rounded-lg">
                  <ClockIcon className="w-6 h-6 text-purple-600" />
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Execution History
                  </h3>
                  <p className="text-sm text-gray-600">View execution logs</p>
                </div>
              </div>
            </Link>
          </div>

          {/* Tools Grid */}
          <div className="bg-white rounded-lg shadow-sm">
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">
                  Available Tools
                </h2>
                <div className="flex items-center space-x-2">
                  <FunnelIcon className="w-5 h-5 text-gray-400" />
                  <span className="text-sm text-gray-600">
                    {filteredTools.length} of {tools.length} tools
                  </span>
                </div>
              </div>
            </div>

            <div className="p-6">
              {filteredTools.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {filteredTools.map((tool) => (
                    <div
                      key={tool.id}
                      className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 hover:shadow-sm transition-all"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-gray-100 rounded-lg">
                            <WrenchScrewdriverIcon className="w-5 h-5 text-gray-600" />
                          </div>
                          <div>
                            <h3 className="font-semibold text-gray-900">
                              {tool.name}
                            </h3>
                            {tool.category && (
                              <Badge variant="secondary" className="mt-1">
                                {tool.category}
                              </Badge>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          {getToolStatusIcon(tool.status)}
                          <button
                            onClick={() => executeTools(tool.id, {})}
                            className="bg-blue-600 text-white px-3 py-1 rounded-md text-sm hover:bg-blue-700 transition-colors"
                            disabled={tool.status !== 'available'}
                          >
                            Run
                          </button>
                        </div>
                      </div>

                      <p className="text-gray-600 text-sm mb-3">
                        {tool.description}
                      </p>

                      {tool.parameters && tool.parameters.length > 0 && (
                        <div className="mb-3">
                          <h4 className="text-sm font-medium text-gray-700 mb-2">
                            Parameters:
                          </h4>
                          <div className="flex flex-wrap gap-1">
                            {tool.parameters.map((param, index) => (
                              <Badge key={index} variant="outline" className="text-xs">
                                {param.name}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>
                          Used {tool.usage || 0} times
                        </span>
                        {tool.lastUsed && (
                          <span>
                            Last used: {new Date(tool.lastUsed).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <WrenchScrewdriverIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    No tools found
                  </h3>
                  <p className="text-gray-600">
                    {searchQuery || selectedCategory !== 'all'
                      ? 'Try adjusting your search or filter criteria.'
                      : 'No MCP tools are currently available.'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ToolsPage;