import React, { useState, useEffect } from 'react';
import { SDRStorageVisualization } from './SDRStorageVisualization';
import { KnowledgeGraphTreemap } from './KnowledgeGraphTreemap';
import { ZeroCopyMonitor } from './ZeroCopyMonitor';
import { MemoryFlowVisualization } from './MemoryFlowVisualization';
import { CognitiveLayerMemoryVisualization } from './CognitiveLayerMemory';
import {
  SDRStorage,
  KnowledgeGraphMemory,
  ZeroCopyMetrics,
  MemoryFlow,
  CognitiveLayerMemory,
  MemoryPressure
} from '../types/memory';

interface MemoryDashboardProps {
  wsUrl?: string;
  className?: string;
}

export function MemoryDashboard({ wsUrl = 'ws://localhost:8080', className = '' }: MemoryDashboardProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'sdr' | 'knowledge' | 'zerocopy' | 'flow' | 'cognitive'>('overview');
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Memory data states
  const [sdrStorage, setSDRStorage] = useState<SDRStorage>(generateMockSDRStorage());
  const [knowledgeGraphMemory, setKnowledgeGraphMemory] = useState<KnowledgeGraphMemory>(generateMockKnowledgeGraphMemory());
  const [zeroCopyMetrics, setZeroCopyMetrics] = useState<ZeroCopyMetrics>(generateMockZeroCopyMetrics());
  const [zeroCopyHistory, setZeroCopyHistory] = useState<ZeroCopyMetrics[]>([generateMockZeroCopyMetrics()]);
  const [memoryFlows, setMemoryFlows] = useState<MemoryFlow[]>(generateMockMemoryFlows());
  const [cognitiveMemory, setCognitiveMemory] = useState<CognitiveLayerMemory>(generateMockCognitiveMemory());
  const [memoryPressure, setMemoryPressure] = useState<MemoryPressure>(generateMockMemoryPressure());

  // WebSocket connection
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          setIsConnected(true);
          console.log('Connected to memory monitoring WebSocket');
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleMemoryUpdate(data);
            setLastUpdate(new Date());
          } catch (error) {
            console.error('Error parsing memory data:', error);
          }
        };

        ws.onclose = () => {
          setIsConnected(false);
          console.log('Disconnected from memory monitoring WebSocket');
          reconnectTimer = window.setTimeout(connect, 5000);
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };
      } catch (error) {
        console.error('Failed to connect:', error);
        reconnectTimer = window.setTimeout(connect, 5000);
      }
    };

    connect();

    // Simulate updates if not connected
    const simulationInterval = setInterval(() => {
      if (!isConnected) {
        simulateMemoryUpdates();
      }
    }, 2000);

    return () => {
      if (ws) {
        ws.close();
      }
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      clearInterval(simulationInterval);
    };
  }, [wsUrl, isConnected]);

  const handleMemoryUpdate = (data: any) => {
    if (data.sdrStorage) setSDRStorage(data.sdrStorage);
    if (data.knowledgeGraphMemory) setKnowledgeGraphMemory(data.knowledgeGraphMemory);
    if (data.zeroCopyMetrics) {
      setZeroCopyMetrics(data.zeroCopyMetrics);
      setZeroCopyHistory(prev => [...prev.slice(-49), data.zeroCopyMetrics]);
    }
    if (data.memoryFlows) setMemoryFlows(prev => [...prev.slice(-50), ...data.memoryFlows]);
    if (data.cognitiveMemory) setCognitiveMemory(data.cognitiveMemory);
    if (data.memoryPressure) setMemoryPressure(data.memoryPressure);
  };

  const simulateMemoryUpdates = () => {
    // Simulate SDR storage changes
    setSDRStorage(prev => ({
      ...prev,
      activeSDRs: prev.activeSDRs + Math.floor(Math.random() * 10 - 5),
      totalMemoryBytes: prev.totalMemoryBytes + Math.floor(Math.random() * 1000000 - 500000),
      fragmentationLevel: Math.max(0, Math.min(1, prev.fragmentationLevel + (Math.random() * 0.02 - 0.01)))
    }));

    // Simulate zero-copy metrics
    const newZeroCopyMetrics = {
      ...zeroCopyMetrics,
      totalOperations: zeroCopyMetrics.totalOperations + Math.floor(Math.random() * 100),
      savedBytes: zeroCopyMetrics.savedBytes + Math.floor(Math.random() * 1000000),
      copyOnWriteEvents: zeroCopyMetrics.copyOnWriteEvents + Math.floor(Math.random() * 5)
    };
    setZeroCopyMetrics(newZeroCopyMetrics);
    setZeroCopyHistory(prev => [...prev.slice(-49), newZeroCopyMetrics]);

    // Simulate memory flows
    const newFlow: MemoryFlow = {
      timestamp: Date.now(),
      source: ['cortical', 'subcortical', 'cache', 'index'][Math.floor(Math.random() * 4)],
      target: ['cortical', 'subcortical', 'cache', 'embedding'][Math.floor(Math.random() * 4)],
      bytes: Math.floor(Math.random() * 1000000),
      operation: ['allocate', 'free', 'copy', 'share'][Math.floor(Math.random() * 4)] as any,
      duration: Math.random() * 100
    };
    setMemoryFlows(prev => [...prev.slice(-99), newFlow]);
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'sdr', label: 'SDR Storage', icon: '🧠' },
    { id: 'knowledge', label: 'Knowledge Graph', icon: '🌐' },
    { id: 'zerocopy', label: 'Zero-Copy', icon: '⚡' },
    { id: 'flow', label: 'Memory Flow', icon: '🌊' },
    { id: 'cognitive', label: 'Cognitive Layers', icon: '🎯' }
  ];

  return (
    <div className={`min-h-screen bg-gray-950 text-gray-100 ${className}`}>
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold">Memory & Storage Monitoring</h1>
            <p className="text-sm text-gray-400 mt-1">Real-time memory analysis and optimization</p>
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
        {activeTab === 'overview' && (
          <OverviewTab
            sdrStorage={sdrStorage}
            knowledgeGraphMemory={knowledgeGraphMemory}
            zeroCopyMetrics={zeroCopyMetrics}
            cognitiveMemory={cognitiveMemory}
            memoryPressure={memoryPressure}
          />
        )}
        
        {activeTab === 'sdr' && (
          <SDRStorageVisualization storage={sdrStorage} />
        )}
        
        {activeTab === 'knowledge' && (
          <KnowledgeGraphTreemap memory={knowledgeGraphMemory} />
        )}
        
        {activeTab === 'zerocopy' && (
          <ZeroCopyMonitor metrics={zeroCopyMetrics} history={zeroCopyHistory} />
        )}
        
        {activeTab === 'flow' && (
          <MemoryFlowVisualization flows={memoryFlows} />
        )}
        
        {activeTab === 'cognitive' && (
          <CognitiveLayerMemoryVisualization memory={cognitiveMemory} />
        )}
      </div>
    </div>
  );
}

// Overview Tab Component
interface OverviewTabProps {
  sdrStorage: SDRStorage;
  knowledgeGraphMemory: KnowledgeGraphMemory;
  zeroCopyMetrics: ZeroCopyMetrics;
  cognitiveMemory: CognitiveLayerMemory;
  memoryPressure: MemoryPressure;
}

function OverviewTab({
  sdrStorage,
  knowledgeGraphMemory,
  zeroCopyMetrics,
  cognitiveMemory,
  memoryPressure
}: OverviewTabProps) {
  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(2)} ${units[unitIndex]}`;
  };

  const totalMemory = 
    sdrStorage.totalMemoryBytes +
    Object.values(knowledgeGraphMemory).reduce((sum, block) => sum + block.size, 0) +
    cognitiveMemory.subcortical.total +
    cognitiveMemory.cortical.total +
    cognitiveMemory.workingMemory.capacity;

  return (
    <div className="space-y-6">
      {/* System Memory Overview */}
      <div className="bg-gray-900 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4 text-white">System Memory Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800 rounded p-4">
            <div className="text-gray-400 text-sm">Total Memory</div>
            <div className="text-2xl font-bold text-white">{formatBytes(totalMemory)}</div>
          </div>
          <div className="bg-gray-800 rounded p-4">
            <div className="text-gray-400 text-sm">Memory Pressure</div>
            <div className={`text-2xl font-bold ${
              memoryPressure.level === 'critical' ? 'text-red-400' :
              memoryPressure.level === 'high' ? 'text-orange-400' :
              memoryPressure.level === 'medium' ? 'text-yellow-400' :
              'text-green-400'
            }`}>
              {memoryPressure.level.toUpperCase()}
            </div>
          </div>
          <div className="bg-gray-800 rounded p-4">
            <div className="text-gray-400 text-sm">Zero-Copy Savings</div>
            <div className="text-2xl font-bold text-green-400">{formatBytes(zeroCopyMetrics.savedBytes)}</div>
          </div>
          <div className="bg-gray-800 rounded p-4">
            <div className="text-gray-400 text-sm">Fragmentation</div>
            <div className="text-2xl font-bold text-orange-400">{(sdrStorage.fragmentationLevel * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <QuickStatCard
          title="SDR Storage"
          stats={[
            { label: 'Active SDRs', value: sdrStorage.activeSDRs.toLocaleString() },
            { label: 'Compression', value: `${(sdrStorage.compressionRatio * 100).toFixed(1)}%` },
            { label: 'Avg Sparsity', value: `${(sdrStorage.averageSparsity * 100).toFixed(1)}%` }
          ]}
        />
        <QuickStatCard
          title="Knowledge Graph"
          stats={[
            { label: 'Entities', value: formatBytes(knowledgeGraphMemory.entities.used) },
            { label: 'Relations', value: formatBytes(knowledgeGraphMemory.relations.used) },
            { label: 'Cache Hit Rate', value: '87.3%' }
          ]}
        />
        <QuickStatCard
          title="Cognitive Layers"
          stats={[
            { label: 'Subcortical', value: `${((cognitiveMemory.subcortical.used / cognitiveMemory.subcortical.total) * 100).toFixed(1)}%` },
            { label: 'Cortical', value: `${((cognitiveMemory.cortical.used / cognitiveMemory.cortical.total) * 100).toFixed(1)}%` },
            { label: 'Working Memory', value: `${cognitiveMemory.workingMemory.buffers.length} buffers` }
          ]}
        />
      </div>

      {/* Memory Pressure Recommendations */}
      {memoryPressure.recommendations.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4">
          <h4 className="text-lg font-medium text-yellow-400 mb-2">Memory Optimization Recommendations</h4>
          <ul className="space-y-1">
            {memoryPressure.recommendations.map((rec, i) => (
              <li key={i} className="text-sm text-gray-300 flex items-center">
                <span className="mr-2">•</span>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

interface QuickStatCardProps {
  title: string;
  stats: { label: string; value: string }[];
}

function QuickStatCard({ title, stats }: QuickStatCardProps) {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h4 className="text-lg font-medium text-white mb-3">{title}</h4>
      <div className="space-y-2">
        {stats.map((stat, i) => (
          <div key={i} className="flex justify-between items-center">
            <span className="text-sm text-gray-400">{stat.label}</span>
            <span className="text-sm font-medium text-white">{stat.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Mock data generators
function generateMockSDRStorage(): SDRStorage {
  return {
    totalSDRs: 125000,
    activeSDRs: 45000,
    archivedSDRs: 80000,
    totalMemoryBytes: 2147483648, // 2GB
    averageSparsity: 0.02,
    compressionRatio: 0.65,
    fragmentationLevel: 0.15,
    storageBlocks: Array.from({ length: 20 }, (_, i) => ({
      id: `block-${i}`,
      size: 134217728, // 128MB
      used: Math.floor(Math.random() * 134217728),
      fragmented: Math.floor(Math.random() * 13421772),
      patterns: Math.floor(Math.random() * 5000),
      lastAccess: Date.now() - Math.floor(Math.random() * 3600000),
      compressionType: ['overlap', 'dictionary', 'none'][Math.floor(Math.random() * 3)] as any
    }))
  };
}

function generateMockKnowledgeGraphMemory(): KnowledgeGraphMemory {
  return {
    entities: {
      name: 'Entities',
      size: 1073741824, // 1GB
      used: 805306368,
      metadata: {
        accessCount: 1250000,
        lastAccess: Date.now(),
        fragmentation: 0.08
      }
    },
    relations: {
      name: 'Relations',
      size: 536870912, // 512MB
      used: 402653184,
      metadata: {
        accessCount: 890000,
        lastAccess: Date.now() - 1000,
        fragmentation: 0.12
      }
    },
    embeddings: {
      name: 'Embeddings',
      size: 2147483648, // 2GB
      used: 1610612736,
      metadata: {
        accessCount: 450000,
        lastAccess: Date.now() - 2000,
        fragmentation: 0.05
      }
    },
    indexes: {
      name: 'Indexes',
      size: 268435456, // 256MB
      used: 201326592,
      metadata: {
        accessCount: 2100000,
        lastAccess: Date.now(),
        fragmentation: 0.18
      }
    },
    cache: {
      name: 'Cache',
      size: 1073741824, // 1GB
      used: 536870912,
      metadata: {
        accessCount: 5400000,
        lastAccess: Date.now(),
        fragmentation: 0.25
      }
    }
  };
}

function generateMockZeroCopyMetrics(): ZeroCopyMetrics {
  return {
    enabled: true,
    totalOperations: 1250000,
    savedBytes: 5368709120, // 5GB
    copyOnWriteEvents: 12500,
    sharedRegions: 856,
    efficiency: 0.85
  };
}

function generateMockMemoryFlows(): MemoryFlow[] {
  const nodes = ['cortical', 'subcortical', 'cache', 'index', 'embedding', 'working'];
  const operations: Array<'allocate' | 'free' | 'copy' | 'share'> = ['allocate', 'free', 'copy', 'share'];
  
  return Array.from({ length: 50 }, () => ({
    timestamp: Date.now() - Math.floor(Math.random() * 60000),
    source: nodes[Math.floor(Math.random() * nodes.length)],
    target: nodes[Math.floor(Math.random() * nodes.length)],
    bytes: Math.floor(Math.random() * 10485760), // Up to 10MB
    operation: operations[Math.floor(Math.random() * operations.length)],
    duration: Math.random() * 100
  }));
}

function generateMockCognitiveMemory(): CognitiveLayerMemory {
  return {
    subcortical: {
      total: 536870912, // 512MB
      used: 402653184,
      components: {
        thalamus: 134217728,
        hippocampus: 100663296,
        amygdala: 83886080,
        basalGanglia: 83886080
      }
    },
    cortical: {
      total: 2147483648, // 2GB
      used: 1610612736,
      regions: {
        prefrontal: 536870912,
        temporal: 402653184,
        parietal: 335544320,
        occipital: 335544320
      }
    },
    workingMemory: {
      capacity: 67108864, // 64MB
      used: 50331648,
      buffers: Array.from({ length: 8 }, (_, i) => ({
        id: `buffer-${i}`,
        content: `Working memory content ${i}`,
        size: 6291456, // 6MB
        age: Math.floor(Math.random() * 10000),
        accessCount: Math.floor(Math.random() * 100),
        priority: Math.random()
      }))
    }
  };
}

function generateMockMemoryPressure(): MemoryPressure {
  return {
    level: 'medium',
    percentage: 65,
    swapUsed: 134217728, // 128MB
    pageCache: 268435456, // 256MB
    recommendations: [
      'Consider clearing unused embeddings cache',
      'Archive old SDR patterns to reduce active memory',
      'Enable more aggressive garbage collection'
    ]
  };
}