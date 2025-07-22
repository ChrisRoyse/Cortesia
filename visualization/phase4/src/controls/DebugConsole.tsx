/**
 * Interactive debugging and system inspection for LLMKG Phase 4
 * Provides comprehensive debugging tools and real-time system monitoring
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

interface LogEntry {
  id: string;
  timestamp: Date;
  level: 'debug' | 'info' | 'warn' | 'error';
  category: 'system' | 'websocket' | 'cognitive' | 'memory' | 'rendering' | 'user';
  message: string;
  data?: any;
  source?: string;
  stackTrace?: string;
}

interface SystemStatus {
  websocket: {
    connected: boolean;
    url: string;
    lastMessage: Date | null;
    messageCount: number;
    errorCount: number;
  };
  cognitive: {
    activePatterns: string[];
    processingQueue: number;
    lastActivation: Date | null;
    inhibitionActive: boolean;
  };
  memory: {
    operations: {
      store: number;
      retrieve: number;
      update: number;
      delete: number;
    };
    cacheHitRate: number;
    memoryPressure: 'low' | 'medium' | 'high';
  };
  rendering: {
    fps: number;
    frameTime: number;
    activeComponents: number;
    memoryUsage: number;
  };
}

interface DebugCommand {
  command: string;
  description: string;
  usage: string;
  handler: (args: string[]) => Promise<any>;
}

interface DataInspector {
  path: string[];
  data: any;
  expanded: Set<string>;
}

interface DebugConsoleProps {
  isVisible: boolean;
  onClose: () => void;
  onCommandExecute: (command: string, args: string[]) => Promise<any>;
  systemData: any;
}

const DebugConsole: React.FC<DebugConsoleProps> = ({
  isVisible,
  onClose,
  onCommandExecute,
  systemData
}) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    websocket: { connected: false, url: '', lastMessage: null, messageCount: 0, errorCount: 0 },
    cognitive: { activePatterns: [], processingQueue: 0, lastActivation: null, inhibitionActive: false },
    memory: { operations: { store: 0, retrieve: 0, update: 0, delete: 0 }, cacheHitRate: 0, memoryPressure: 'low' },
    rendering: { fps: 60, frameTime: 16.67, activeComponents: 0, memoryUsage: 0 }
  });

  const [activeTab, setActiveTab] = useState<'logs' | 'status' | 'inspector' | 'console'>('logs');
  const [logFilter, setLogFilter] = useState({
    level: 'all' as 'all' | LogEntry['level'],
    category: 'all' as 'all' | LogEntry['category'],
    search: ''
  });

  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [currentCommand, setCurrentCommand] = useState('');
  const [commandOutput, setCommandOutput] = useState<Array<{ command: string; output: any; timestamp: Date }>>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const [inspector, setInspector] = useState<DataInspector>({
    path: [],
    data: systemData,
    expanded: new Set(['root'])
  });

  const logsEndRef = useRef<HTMLDivElement>(null);
  const commandInputRef = useRef<HTMLInputElement>(null);

  // Debug commands
  const debugCommands: DebugCommand[] = [
    {
      command: 'help',
      description: 'Show available commands',
      usage: 'help [command]',
      handler: async (args) => {
        if (args.length > 0) {
          const cmd = debugCommands.find(c => c.command === args[0]);
          return cmd ? `${cmd.command}: ${cmd.description}\nUsage: ${cmd.usage}` : `Unknown command: ${args[0]}`;
        }
        return debugCommands.map(cmd => `${cmd.command.padEnd(15)} ${cmd.description}`).join('\n');
      }
    },
    {
      command: 'clear',
      description: 'Clear console output',
      usage: 'clear',
      handler: async () => {
        setCommandOutput([]);
        return 'Console cleared';
      }
    },
    {
      command: 'status',
      description: 'Show system status',
      usage: 'status [component]',
      handler: async (args) => {
        if (args.length > 0) {
          const component = args[0] as keyof SystemStatus;
          if (component in systemStatus) {
            return JSON.stringify(systemStatus[component], null, 2);
          }
          return `Unknown component: ${args[0]}`;
        }
        return JSON.stringify(systemStatus, null, 2);
      }
    },
    {
      command: 'logs',
      description: 'Show recent logs',
      usage: 'logs [count] [level] [category]',
      handler: async (args) => {
        const count = args.length > 0 ? parseInt(args[0]) || 10 : 10;
        const level = args.length > 1 ? args[1] as LogEntry['level'] : undefined;
        const category = args.length > 2 ? args[2] as LogEntry['category'] : undefined;
        
        let filtered = logs;
        if (level) filtered = filtered.filter(log => log.level === level);
        if (category) filtered = filtered.filter(log => log.category === category);
        
        return filtered
          .slice(-count)
          .map(log => `[${log.timestamp.toISOString()}] ${log.level.toUpperCase()} (${log.category}): ${log.message}`)
          .join('\n');
      }
    },
    {
      command: 'inspect',
      description: 'Inspect data object',
      usage: 'inspect <path>',
      handler: async (args) => {
        if (args.length === 0) {
          return 'Usage: inspect <path>';
        }
        
        const path = args[0].split('.');
        let data = systemData;
        
        for (const key of path) {
          if (data && typeof data === 'object' && key in data) {
            data = data[key];
          } else {
            return `Path not found: ${args[0]}`;
          }
        }
        
        return JSON.stringify(data, null, 2);
      }
    },
    {
      command: 'memory',
      description: 'Show memory usage information',
      usage: 'memory',
      handler: async () => {
        if ('memory' in performance) {
          const memory = (performance as any).memory;
          return {
            used: `${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
            total: `${(memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
            limit: `${(memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`
          };
        }
        return 'Memory information not available';
      }
    },
    {
      command: 'websocket',
      description: 'WebSocket connection commands',
      usage: 'websocket <connect|disconnect|status|send> [data]',
      handler: async (args) => {
        if (args.length === 0) {
          return 'Usage: websocket <connect|disconnect|status|send> [data]';
        }
        
        const action = args[0];
        switch (action) {
          case 'status':
            return systemStatus.websocket;
          case 'connect':
            return 'WebSocket connection initiated';
          case 'disconnect':
            return 'WebSocket disconnected';
          case 'send':
            if (args.length < 2) {
              return 'Usage: websocket send <data>';
            }
            return `Sent: ${args.slice(1).join(' ')}`;
          default:
            return `Unknown websocket action: ${action}`;
        }
      }
    },
    {
      command: 'cognitive',
      description: 'Cognitive pattern debugging',
      usage: 'cognitive <patterns|activate|inhibit|status>',
      handler: async (args) => {
        if (args.length === 0) {
          return 'Usage: cognitive <patterns|activate|inhibit|status>';
        }
        
        const action = args[0];
        switch (action) {
          case 'patterns':
            return systemStatus.cognitive.activePatterns.join(', ') || 'No active patterns';
          case 'status':
            return systemStatus.cognitive;
          case 'activate':
            return 'Pattern activation simulated';
          case 'inhibit':
            return 'Pattern inhibition simulated';
          default:
            return `Unknown cognitive action: ${action}`;
        }
      }
    }
  ];

  // Initialize console
  useEffect(() => {
    if (isVisible) {
      startSystemMonitoring();
      addLog('info', 'system', 'Debug console initialized');
    }

    return () => {
      stopSystemMonitoring();
    };
  }, [isVisible]);

  // Filter logs
  useEffect(() => {
    let filtered = logs;

    if (logFilter.level !== 'all') {
      filtered = filtered.filter(log => log.level === logFilter.level);
    }

    if (logFilter.category !== 'all') {
      filtered = filtered.filter(log => log.category === logFilter.category);
    }

    if (logFilter.search) {
      const search = logFilter.search.toLowerCase();
      filtered = filtered.filter(log => 
        log.message.toLowerCase().includes(search) ||
        log.source?.toLowerCase().includes(search)
      );
    }

    setFilteredLogs(filtered);
  }, [logs, logFilter]);

  // Scroll to bottom when new logs arrive
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [filteredLogs]);

  // System monitoring
  const startSystemMonitoring = useCallback(() => {
    const interval = setInterval(() => {
      updateSystemStatus();
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const stopSystemMonitoring = useCallback(() => {
    // Cleanup monitoring
  }, []);

  const updateSystemStatus = useCallback(() => {
    // Mock system status updates - would integrate with actual system
    setSystemStatus(prev => ({
      websocket: {
        ...prev.websocket,
        connected: Math.random() > 0.1, // 90% uptime simulation
        messageCount: prev.websocket.messageCount + Math.floor(Math.random() * 3),
        lastMessage: Math.random() > 0.7 ? new Date() : prev.websocket.lastMessage
      },
      cognitive: {
        ...prev.cognitive,
        activePatterns: ['ConvergentThinking', 'AnalyticalReasoning'].filter(() => Math.random() > 0.5),
        processingQueue: Math.floor(Math.random() * 10),
        lastActivation: Math.random() > 0.8 ? new Date() : prev.cognitive.lastActivation,
        inhibitionActive: Math.random() > 0.7
      },
      memory: {
        operations: {
          store: prev.memory.operations.store + Math.floor(Math.random() * 5),
          retrieve: prev.memory.operations.retrieve + Math.floor(Math.random() * 10),
          update: prev.memory.operations.update + Math.floor(Math.random() * 3),
          delete: prev.memory.operations.delete + Math.floor(Math.random() * 2)
        },
        cacheHitRate: 75 + Math.random() * 20, // 75-95%
        memoryPressure: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as any
      },
      rendering: {
        fps: 55 + Math.random() * 10, // 55-65 FPS
        frameTime: 15 + Math.random() * 5, // 15-20ms
        activeComponents: Math.floor(Math.random() * 20) + 10,
        memoryUsage: Math.random() * 100 + 50 // 50-150MB
      }
    }));

    // Generate random log entries
    if (Math.random() > 0.7) {
      const categories: LogEntry['category'][] = ['system', 'websocket', 'cognitive', 'memory', 'rendering'];
      const levels: LogEntry['level'][] = ['debug', 'info', 'warn', 'error'];
      const messages = [
        'Pattern activation completed',
        'WebSocket message received',
        'Memory operation executed',
        'Rendering frame completed',
        'Cache hit ratio updated',
        'System heartbeat',
        'Performance threshold exceeded',
        'Data validation passed'
      ];

      addLog(
        levels[Math.floor(Math.random() * levels.length)],
        categories[Math.floor(Math.random() * categories.length)],
        messages[Math.floor(Math.random() * messages.length)]
      );
    }
  }, []);

  // Logging functions
  const addLog = useCallback((level: LogEntry['level'], category: LogEntry['category'], message: string, data?: any) => {
    const entry: LogEntry = {
      id: crypto.randomUUID(),
      timestamp: new Date(),
      level,
      category,
      message,
      data,
      source: `Debug Console`
    };

    setLogs(prev => [...prev.slice(-999), entry]); // Keep last 1000 logs
  }, []);

  // Command execution
  const executeCommand = useCallback(async (command: string) => {
    if (!command.trim()) return;

    const [cmd, ...args] = command.trim().split(/\s+/);
    const debugCmd = debugCommands.find(c => c.command === cmd);

    let output: any;
    const timestamp = new Date();

    try {
      if (debugCmd) {
        output = await debugCmd.handler(args);
      } else {
        // Try external command handler
        output = await onCommandExecute(cmd, args);
      }
    } catch (error) {
      output = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
      addLog('error', 'system', `Command failed: ${command}`, { error });
    }

    setCommandOutput(prev => [...prev, { command, output, timestamp }]);
    setCommandHistory(prev => [...prev.slice(-49), command]); // Keep last 50 commands
    setCurrentCommand('');
    setHistoryIndex(-1);

    addLog('debug', 'user', `Command executed: ${command}`, { output });
  }, [debugCommands, onCommandExecute, addLog]);

  // Command input handling
  const handleCommandKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'Enter':
        e.preventDefault();
        executeCommand(currentCommand);
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (commandHistory.length > 0) {
          const newIndex = Math.min(historyIndex + 1, commandHistory.length - 1);
          setHistoryIndex(newIndex);
          setCurrentCommand(commandHistory[commandHistory.length - 1 - newIndex]);
        }
        break;
      case 'ArrowDown':
        e.preventDefault();
        if (historyIndex > 0) {
          const newIndex = historyIndex - 1;
          setHistoryIndex(newIndex);
          setCurrentCommand(commandHistory[commandHistory.length - 1 - newIndex]);
        } else if (historyIndex === 0) {
          setHistoryIndex(-1);
          setCurrentCommand('');
        }
        break;
      case 'Tab':
        e.preventDefault();
        // Auto-complete
        const partial = currentCommand.toLowerCase();
        const matches = debugCommands.filter(cmd => cmd.command.startsWith(partial));
        if (matches.length === 1) {
          setCurrentCommand(matches[0].command);
        }
        break;
    }
  }, [currentCommand, commandHistory, historyIndex, executeCommand]);

  // Data inspector
  const toggleInspectorNode = useCallback((path: string) => {
    setInspector(prev => {
      const newExpanded = new Set(prev.expanded);
      if (newExpanded.has(path)) {
        newExpanded.delete(path);
      } else {
        newExpanded.add(path);
      }
      return { ...prev, expanded: newExpanded };
    });
  }, []);

  const renderInspectorNode = (data: any, path: string[], level: number = 0): JSX.Element[] => {
    const fullPath = path.join('.');
    const isExpanded = inspector.expanded.has(fullPath);
    const hasChildren = data && typeof data === 'object' && Object.keys(data).length > 0;

    const elements: JSX.Element[] = [];

    // Current node
    elements.push(
      <div
        key={fullPath}
        className={`flex items-center py-1 px-2 hover:bg-gray-100 cursor-pointer`}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={() => hasChildren && toggleInspectorNode(fullPath)}
      >
        {hasChildren && (
          <span className="mr-2 text-xs">
            {isExpanded ? '▼' : '▶'}
          </span>
        )}
        <span className="font-mono text-sm">
          <span className="text-blue-600">{path[path.length - 1] || 'root'}</span>
          <span className="text-gray-500">: </span>
          <span className="text-gray-800">
            {typeof data === 'object' ? 
              Array.isArray(data) ? `Array[${data.length}]` : `Object{${Object.keys(data).length}}` :
              JSON.stringify(data)
            }
          </span>
        </span>
      </div>
    );

    // Children nodes
    if (isExpanded && hasChildren) {
      for (const [key, value] of Object.entries(data)) {
        elements.push(...renderInspectorNode(value, [...path, key], level + 1));
      }
    }

    return elements;
  };

  const formatLogEntry = (log: LogEntry) => {
    const levelColors = {
      debug: 'text-gray-600',
      info: 'text-blue-600',
      warn: 'text-yellow-600',
      error: 'text-red-600'
    };

    return (
      <div key={log.id} className="p-2 border-b border-gray-100 hover:bg-gray-50 text-sm font-mono">
        <div className="flex items-start space-x-2">
          <span className="text-gray-400 text-xs min-w-0 flex-shrink-0">
            {log.timestamp.toLocaleTimeString()}
          </span>
          <span className={`text-xs font-bold min-w-0 flex-shrink-0 ${levelColors[log.level]}`}>
            {log.level.toUpperCase()}
          </span>
          <span className="text-xs text-purple-600 min-w-0 flex-shrink-0">
            {log.category}
          </span>
          <span className="text-gray-800 break-all">
            {log.message}
          </span>
        </div>
        {log.data && (
          <div className="mt-1 ml-8 text-xs text-gray-600 bg-gray-100 p-1 rounded">
            {typeof log.data === 'string' ? log.data : JSON.stringify(log.data, null, 2)}
          </div>
        )}
      </div>
    );
  };

  if (!isVisible) return null;

  return (
    <div className="fixed bottom-4 left-4 w-[800px] h-[600px] bg-white rounded-lg shadow-xl border z-50 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b flex justify-between items-center bg-gray-900 text-white rounded-t-lg">
        <div className="flex items-center space-x-2">
          <h2 className="text-lg font-semibold">Debug Console</h2>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-xs">Active</span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-gray-300 hover:text-white"
        >
          ✕
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b bg-gray-50">
        {([
          { key: 'logs', label: `Logs (${logs.length})` },
          { key: 'status', label: 'Status' },
          { key: 'inspector', label: 'Inspector' },
          { key: 'console', label: 'Console' }
        ] as const).map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`px-4 py-2 text-sm border-b-2 ${
              activeTab === key
                ? 'border-blue-500 text-blue-600 bg-white'
                : 'border-transparent text-gray-600 hover:text-gray-800'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {/* Logs Tab */}
        {activeTab === 'logs' && (
          <div className="h-full flex flex-col">
            {/* Log filters */}
            <div className="p-3 border-b bg-gray-50 flex space-x-4">
              <select
                value={logFilter.level}
                onChange={(e) => setLogFilter(prev => ({ ...prev, level: e.target.value as any }))}
                className="text-sm border rounded px-2 py-1"
              >
                <option value="all">All Levels</option>
                <option value="debug">Debug</option>
                <option value="info">Info</option>
                <option value="warn">Warning</option>
                <option value="error">Error</option>
              </select>

              <select
                value={logFilter.category}
                onChange={(e) => setLogFilter(prev => ({ ...prev, category: e.target.value as any }))}
                className="text-sm border rounded px-2 py-1"
              >
                <option value="all">All Categories</option>
                <option value="system">System</option>
                <option value="websocket">WebSocket</option>
                <option value="cognitive">Cognitive</option>
                <option value="memory">Memory</option>
                <option value="rendering">Rendering</option>
                <option value="user">User</option>
              </select>

              <input
                type="text"
                placeholder="Search logs..."
                value={logFilter.search}
                onChange={(e) => setLogFilter(prev => ({ ...prev, search: e.target.value }))}
                className="text-sm border rounded px-2 py-1 flex-1"
              />

              <button
                onClick={() => setLogs([])}
                className="text-sm px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Clear
              </button>
            </div>

            {/* Log entries */}
            <div className="flex-1 overflow-auto">
              {filteredLogs.length === 0 ? (
                <div className="p-8 text-center text-gray-500">
                  No logs to display
                </div>
              ) : (
                <div>
                  {filteredLogs.map(formatLogEntry)}
                  <div ref={logsEndRef} />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Status Tab */}
        {activeTab === 'status' && (
          <div className="p-4 overflow-auto">
            <div className="grid grid-cols-2 gap-6">
              {/* WebSocket Status */}
              <div className="space-y-3">
                <h3 className="font-semibold text-gray-800 flex items-center">
                  <div className={`w-3 h-3 rounded-full mr-2 ${
                    systemStatus.websocket.connected ? 'bg-green-500' : 'bg-red-500'
                  }`} />
                  WebSocket
                </h3>
                <div className="text-sm space-y-1">
                  <div>Status: {systemStatus.websocket.connected ? 'Connected' : 'Disconnected'}</div>
                  <div>Messages: {systemStatus.websocket.messageCount}</div>
                  <div>Errors: {systemStatus.websocket.errorCount}</div>
                  <div>Last Message: {systemStatus.websocket.lastMessage?.toLocaleTimeString() || 'None'}</div>
                </div>
              </div>

              {/* Cognitive Status */}
              <div className="space-y-3">
                <h3 className="font-semibold text-gray-800 flex items-center">
                  <div className={`w-3 h-3 rounded-full mr-2 ${
                    systemStatus.cognitive.activePatterns.length > 0 ? 'bg-blue-500' : 'bg-gray-400'
                  }`} />
                  Cognitive Patterns
                </h3>
                <div className="text-sm space-y-1">
                  <div>Active: {systemStatus.cognitive.activePatterns.join(', ') || 'None'}</div>
                  <div>Queue: {systemStatus.cognitive.processingQueue}</div>
                  <div>Inhibition: {systemStatus.cognitive.inhibitionActive ? 'Active' : 'Inactive'}</div>
                  <div>Last: {systemStatus.cognitive.lastActivation?.toLocaleTimeString() || 'None'}</div>
                </div>
              </div>

              {/* Memory Status */}
              <div className="space-y-3">
                <h3 className="font-semibold text-gray-800 flex items-center">
                  <div className={`w-3 h-3 rounded-full mr-2 ${
                    systemStatus.memory.memoryPressure === 'low' ? 'bg-green-500' :
                    systemStatus.memory.memoryPressure === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                  }`} />
                  Memory Operations
                </h3>
                <div className="text-sm space-y-1">
                  <div>Store: {systemStatus.memory.operations.store}</div>
                  <div>Retrieve: {systemStatus.memory.operations.retrieve}</div>
                  <div>Update: {systemStatus.memory.operations.update}</div>
                  <div>Delete: {systemStatus.memory.operations.delete}</div>
                  <div>Cache Hit: {systemStatus.memory.cacheHitRate.toFixed(1)}%</div>
                  <div>Pressure: {systemStatus.memory.memoryPressure}</div>
                </div>
              </div>

              {/* Rendering Status */}
              <div className="space-y-3">
                <h3 className="font-semibold text-gray-800 flex items-center">
                  <div className={`w-3 h-3 rounded-full mr-2 ${
                    systemStatus.rendering.fps > 50 ? 'bg-green-500' :
                    systemStatus.rendering.fps > 30 ? 'bg-yellow-500' : 'bg-red-500'
                  }`} />
                  Rendering
                </h3>
                <div className="text-sm space-y-1">
                  <div>FPS: {systemStatus.rendering.fps.toFixed(1)}</div>
                  <div>Frame Time: {systemStatus.rendering.frameTime.toFixed(2)}ms</div>
                  <div>Components: {systemStatus.rendering.activeComponents}</div>
                  <div>Memory: {systemStatus.rendering.memoryUsage.toFixed(1)}MB</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Inspector Tab */}
        {activeTab === 'inspector' && (
          <div className="h-full overflow-auto">
            <div className="p-3 border-b bg-gray-50">
              <h3 className="font-medium">Data Inspector</h3>
              <p className="text-sm text-gray-600">Click to expand/collapse nodes</p>
            </div>
            <div className="text-sm">
              {renderInspectorNode(systemData || {}, [])}
            </div>
          </div>
        )}

        {/* Console Tab */}
        {activeTab === 'console' && (
          <div className="h-full flex flex-col">
            {/* Command output */}
            <div className="flex-1 overflow-auto p-2 bg-gray-900 text-green-400 font-mono text-sm">
              <div className="mb-2">
                LLMKG Debug Console v1.0
                <br />
                Type 'help' for available commands.
              </div>
              
              {commandOutput.map((output, index) => (
                <div key={index} className="mb-2">
                  <div className="text-blue-400">$ {output.command}</div>
                  <div className="text-gray-300 whitespace-pre-wrap ml-2">
                    {typeof output.output === 'string' 
                      ? output.output 
                      : JSON.stringify(output.output, null, 2)
                    }
                  </div>
                </div>
              ))}
            </div>

            {/* Command input */}
            <div className="p-2 border-t bg-gray-800 flex items-center">
              <span className="text-green-400 font-mono mr-2">$</span>
              <input
                ref={commandInputRef}
                type="text"
                value={currentCommand}
                onChange={(e) => setCurrentCommand(e.target.value)}
                onKeyDown={handleCommandKeyDown}
                placeholder="Enter command..."
                className="flex-1 bg-transparent text-green-400 font-mono outline-none"
                autoComplete="off"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DebugConsole;