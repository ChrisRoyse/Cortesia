import React, { useState, useEffect } from 'react';
import { DistributedTracing } from './DistributedTracing';
import { TimeTravelDebugger } from './TimeTravelDebugger';
import { QueryAnalyzer } from './QueryAnalyzer';
import { ErrorLoggingDashboard } from './ErrorLoggingDashboard';
import {
  DistributedTrace,
  TimeTravelSession,
  TimeTravelSnapshot,
  QueryAnalysis,
  ErrorLog,
  ErrorStats
} from '../types/debugging';

interface DebuggingDashboardProps {
  wsUrl?: string;
  className?: string;
}

export function DebuggingDashboard({ 
  wsUrl = 'ws://localhost:8080', 
  className = '' 
}: DebuggingDashboardProps) {
  const [activeTab, setActiveTab] = useState<'tracing' | 'timetravel' | 'query' | 'errors'>('tracing');
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Debugging data states
  const [traces, setTraces] = useState<DistributedTrace[]>(generateMockTraces());
  const [timeTravelSession, setTimeTravelSession] = useState<TimeTravelSession>(generateMockTimeTravelSession());
  const [queryAnalyses, setQueryAnalyses] = useState<QueryAnalysis[]>(generateMockQueryAnalyses());
  const [errorLogs, setErrorLogs] = useState<ErrorLog[]>(generateMockErrorLogs());
  const [errorStats, setErrorStats] = useState<ErrorStats>(generateMockErrorStats());

  // WebSocket connection
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: NodeJS.Timeout | null = null;

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          setIsConnected(true);
          console.log('Connected to debugging WebSocket');
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleDebugUpdate(data);
            setLastUpdate(new Date());
          } catch (error) {
            console.error('Error parsing debug data:', error);
          }
        };

        ws.onclose = () => {
          setIsConnected(false);
          console.log('Disconnected from debugging WebSocket');
          reconnectTimer = setTimeout(connect, 5000);
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };
      } catch (error) {
        console.error('Failed to connect:', error);
        reconnectTimer = setTimeout(connect, 5000);
      }
    };

    connect();

    // Simulate updates if not connected
    const simulationInterval = setInterval(() => {
      if (!isConnected) {
        simulateDebugUpdates();
      }
    }, 5000);

    return () => {
      if (ws) {
        ws.close();
      }
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      clearInterval(simulationInterval);
    };
  }, [wsUrl, isConnected]);

  const handleDebugUpdate = (data: any) => {
    if (data.traces) setTraces(prev => [...data.traces, ...prev].slice(0, 50));
    if (data.snapshot) {
      setTimeTravelSession(prev => ({
        ...prev,
        snapshots: [...prev.snapshots, data.snapshot].slice(-100)
      }));
    }
    if (data.queryAnalysis) setQueryAnalyses(prev => [data.queryAnalysis, ...prev].slice(0, 50));
    if (data.errors) {
      setErrorLogs(prev => [...data.errors, ...prev].slice(0, 100));
      updateErrorStats(data.errors);
    }
  };

  const simulateDebugUpdates = () => {
    // Simulate new trace
    if (Math.random() > 0.7) {
      const newTrace = generateMockTrace();
      setTraces(prev => [newTrace, ...prev].slice(0, 50));
    }

    // Simulate new error
    if (Math.random() > 0.8) {
      const newError = generateMockError();
      setErrorLogs(prev => [newError, ...prev].slice(0, 100));
      updateErrorStats([newError]);
    }
  };

  const updateErrorStats = (newErrors: ErrorLog[]) => {
    setErrorStats(prev => {
      const updated = { ...prev };
      updated.total += newErrors.length;
      
      newErrors.forEach(error => {
        updated.byLevel[error.level] = (updated.byLevel[error.level] || 0) + 1;
        updated.byCategory[error.category] = (updated.byCategory[error.category] || 0) + 1;
        updated.byService[error.context.service] = (updated.byService[error.context.service] || 0) + 1;
      });

      // Update trend
      const latestTimestamp = Math.max(...updated.trend.map(t => t.timestamp), Date.now());
      updated.trend.push({
        timestamp: latestTimestamp,
        count: newErrors.length
      });
      updated.trend = updated.trend.slice(-100);

      return updated;
    });
  };

  const handleSnapshotChange = (snapshot: TimeTravelSnapshot) => {
    console.log('Snapshot changed:', snapshot);
    // Handle snapshot change - could emit events or update other components
  };

  const handleCompareSnapshots = (baseId: string, compareId: string) => {
    console.log('Comparing snapshots:', baseId, compareId);
    // Handle snapshot comparison
  };

  const handleOptimizeQuery = (queryId: string, suggestion: any) => {
    console.log('Optimizing query:', queryId, suggestion);
    // Handle query optimization
  };

  const handleResolveError = (errorId: string) => {
    setErrorLogs(prev => prev.map(error => 
      error.id === errorId ? { ...error, resolved: true } : error
    ));
  };

  const handleFilterErrors = (category: string | null, level: string | null) => {
    console.log('Filtering errors:', { category, level });
    // Handle error filtering
  };

  const tabs = [
    { id: 'tracing', label: 'Distributed Tracing', icon: '🔍' },
    { id: 'timetravel', label: 'Time Travel', icon: '⏰' },
    { id: 'query', label: 'Query Analyzer', icon: '📊' },
    { id: 'errors', label: 'Error Logs', icon: '⚠️' }
  ];

  return (
    <div className={`min-h-screen bg-gray-950 text-gray-100 ${className}`}>
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold">Advanced Debugging Tools</h1>
            <p className="text-sm text-gray-400 mt-1">Deep insights into system behavior and performance</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`flex items-center ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <div className="text-sm text-gray-400">
              Last update: {lastUpdate.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-gray-900 border-b border-gray-800 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="flex space-x-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-4 py-3 text-sm font-medium transition-colors duration-150 border-b-2 ${
                  activeTab === tab.id
                    ? 'text-blue-400 border-blue-400'
                    : 'text-gray-400 border-transparent hover:text-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto p-6">
        {activeTab === 'tracing' && (
          <DistributedTracing traces={traces} />
        )}
        
        {activeTab === 'timetravel' && (
          <TimeTravelDebugger 
            session={timeTravelSession}
            onSnapshotChange={handleSnapshotChange}
            onCompare={handleCompareSnapshots}
          />
        )}
        
        {activeTab === 'query' && (
          <QueryAnalyzer 
            analyses={queryAnalyses}
            onOptimize={handleOptimizeQuery}
          />
        )}
        
        {activeTab === 'errors' && (
          <ErrorLoggingDashboard 
            errors={errorLogs}
            stats={errorStats}
            onResolve={handleResolveError}
            onFilter={handleFilterErrors}
          />
        )}

        {/* Quick Stats Bar */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-900 rounded p-4">
            <div className="text-sm text-gray-400">Active Traces</div>
            <div className="text-2xl font-bold text-white">{traces.length}</div>
          </div>
          <div className="bg-gray-900 rounded p-4">
            <div className="text-sm text-gray-400">Snapshots</div>
            <div className="text-2xl font-bold text-blue-400">{timeTravelSession.snapshots.length}</div>
          </div>
          <div className="bg-gray-900 rounded p-4">
            <div className="text-sm text-gray-400">Slow Queries</div>
            <div className="text-2xl font-bold text-orange-400">
              {queryAnalyses.filter(q => q.executionTime > 1000).length}
            </div>
          </div>
          <div className="bg-gray-900 rounded p-4">
            <div className="text-sm text-gray-400">Unresolved Errors</div>
            <div className="text-2xl font-bold text-red-400">
              {errorLogs.filter(e => !e.resolved).length}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Mock data generators
function generateMockTrace(): DistributedTrace {
  const services = ['api-gateway', 'auth-service', 'cognitive-engine', 'data-store', 'cache'];
  const operations = ['query', 'authenticate', 'process', 'fetch', 'store'];
  
  const traceId = `trace-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const startTime = Date.now() - Math.random() * 10000;
  
  const spans: any[] = [];
  let currentTime = startTime;
  
  // Root span
  const rootSpan = {
    traceId,
    spanId: `span-${Math.random().toString(36).substr(2, 9)}`,
    operationName: operations[0],
    serviceName: services[0],
    startTime: currentTime,
    endTime: currentTime + Math.random() * 500,
    duration: 0,
    status: Math.random() > 0.1 ? 'success' : 'error',
    tags: { userId: 'user123', requestId: `req-${Date.now()}` },
    logs: [],
    references: []
  };
  rootSpan.duration = rootSpan.endTime - rootSpan.startTime;
  spans.push(rootSpan);
  
  // Child spans
  for (let i = 0; i < Math.floor(Math.random() * 10) + 5; i++) {
    currentTime += Math.random() * 50;
    const span = {
      traceId,
      spanId: `span-${Math.random().toString(36).substr(2, 9)}`,
      parentSpanId: i === 0 ? rootSpan.spanId : spans[Math.floor(Math.random() * spans.length)].spanId,
      operationName: operations[Math.floor(Math.random() * operations.length)],
      serviceName: services[Math.floor(Math.random() * services.length)],
      startTime: currentTime,
      endTime: currentTime + Math.random() * 200,
      duration: 0,
      status: Math.random() > 0.2 ? 'success' : Math.random() > 0.5 ? 'warning' : 'error',
      tags: {},
      logs: Math.random() > 0.7 ? [{
        timestamp: currentTime + Math.random() * 100,
        level: 'info',
        message: 'Processing request',
        fields: {}
      }] : [],
      references: []
    };
    span.duration = span.endTime - span.startTime;
    spans.push(span);
  }
  
  return {
    traceId,
    spans,
    rootSpan,
    services: Array.from(new Set(spans.map(s => s.serviceName))),
    startTime,
    endTime: Math.max(...spans.map(s => s.endTime)),
    duration: Math.max(...spans.map(s => s.endTime)) - startTime,
    spanCount: spans.length,
    errorCount: spans.filter(s => s.status === 'error').length
  };
}

function generateMockTraces(): DistributedTrace[] {
  return Array.from({ length: 20 }, () => generateMockTrace());
}

function generateMockTimeTravelSession(): TimeTravelSession {
  const snapshots = Array.from({ length: 50 }, (_, i) => ({
    id: `snapshot-${i}`,
    timestamp: Date.now() - (50 - i) * 60000,
    label: i % 10 === 0 ? `Checkpoint ${i/10}` : `Auto-save ${i}`,
    state: {
      patterns: [],
      connections: [],
      memory: {},
      activations: {}
    },
    metadata: {
      trigger: i % 10 === 0 ? 'manual' : 'auto',
      changes: [`Change ${i}.1`, `Change ${i}.2`],
      performance: {
        cpu: Math.random() * 100,
        memory: Math.random() * 100
      }
    }
  }));
  
  return {
    sessionId: 'session-123',
    snapshots,
    currentIndex: snapshots.length - 1,
    playbackSpeed: 1,
    isPlaying: false
  };
}

function generateMockQueryAnalyses(): QueryAnalysis[] {
  const queries = [
    'SELECT * FROM patterns WHERE activation > 0.5',
    'SELECT p.*, c.* FROM patterns p JOIN connections c ON p.id = c.source_id',
    'UPDATE activations SET value = value * 0.9 WHERE timestamp < NOW() - INTERVAL 1 HOUR',
    'INSERT INTO logs (timestamp, level, message) VALUES ($1, $2, $3)'
  ];
  
  return Array.from({ length: 20 }, (_, i) => ({
    queryId: `query-${i}`,
    query: queries[i % queries.length],
    timestamp: Date.now() - Math.random() * 3600000,
    executionTime: Math.random() * 2000,
    plan: {
      nodes: [{
        id: 'node-1',
        type: 'scan',
        operation: 'Sequential Scan',
        cost: Math.random() * 1000,
        rows: Math.floor(Math.random() * 10000),
        width: 100,
        children: [],
        details: { table: 'patterns' }
      }],
      estimatedCost: Math.random() * 1000,
      estimatedRows: Math.floor(Math.random() * 10000)
    },
    profile: {
      actualTime: Math.random() * 2000,
      planningTime: Math.random() * 50,
      executionTime: Math.random() * 1950,
      rowsProcessed: Math.floor(Math.random() * 10000),
      bytesProcessed: Math.floor(Math.random() * 1000000),
      memoryUsed: Math.floor(Math.random() * 10000000),
      cacheHits: Math.floor(Math.random() * 1000),
      cacheMisses: Math.floor(Math.random() * 100)
    },
    suggestions: Math.random() > 0.5 ? [{
      type: 'index',
      priority: 'high',
      description: 'Create index on activation column',
      impact: 'Could reduce query time by 80%',
      implementation: 'CREATE INDEX idx_patterns_activation ON patterns(activation);'
    }] : [],
    bottlenecks: [{
      component: 'Sequential Scan',
      operation: 'patterns table scan',
      duration: Math.random() * 1000,
      percentage: Math.random() * 100,
      cause: 'Missing index on activation column'
    }]
  }));
}

function generateMockError(): ErrorLog {
  const categories = ['Database', 'Network', 'Authentication', 'Processing', 'Validation'];
  const services = ['api-gateway', 'auth-service', 'cognitive-engine', 'data-store'];
  const operations = ['query', 'connect', 'process', 'validate'];
  
  return {
    id: `error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    timestamp: Date.now(),
    level: Math.random() > 0.8 ? 'critical' : Math.random() > 0.5 ? 'error' : 'warning',
    category: categories[Math.floor(Math.random() * categories.length)],
    message: `Failed to ${operations[Math.floor(Math.random() * operations.length)]} due to timeout`,
    stack: Math.random() > 0.5 ? `Error: Timeout exceeded
  at processRequest (engine.js:123:45)
  at async handleQuery (query.js:56:23)
  at async main (app.js:12:5)` : undefined,
    context: {
      service: services[Math.floor(Math.random() * services.length)],
      operation: operations[Math.floor(Math.random() * operations.length)],
      userId: Math.random() > 0.5 ? 'user123' : undefined,
      requestId: `req-${Date.now()}`,
      metadata: {
        duration: Math.random() * 5000,
        retries: Math.floor(Math.random() * 3)
      }
    },
    frequency: 1,
    firstSeen: Date.now(),
    lastSeen: Date.now(),
    resolved: false
  };
}

function generateMockErrorLogs(): ErrorLog[] {
  return Array.from({ length: 50 }, () => generateMockError());
}

function generateMockErrorStats(): ErrorStats {
  const errors = generateMockErrorLogs();
  const stats: ErrorStats = {
    total: errors.length,
    byLevel: {},
    byCategory: {},
    byService: {},
    trend: [],
    topErrors: []
  };
  
  errors.forEach(error => {
    stats.byLevel[error.level] = (stats.byLevel[error.level] || 0) + 1;
    stats.byCategory[error.category] = (stats.byCategory[error.category] || 0) + 1;
    stats.byService[error.context.service] = (stats.byService[error.context.service] || 0) + 1;
  });
  
  // Generate trend data
  for (let i = 0; i < 24; i++) {
    stats.trend.push({
      timestamp: Date.now() - (24 - i) * 3600000,
      count: Math.floor(Math.random() * 50)
    });
  }
  
  stats.topErrors = errors.slice(0, 5);
  
  return stats;
}