import React, { useState, useEffect } from 'react';
import { PatternActivation3D } from './PatternActivation3D';
import { PatternClassification } from './PatternClassification';
import { InhibitionExcitationBalance } from './InhibitionExcitationBalance';
import { TemporalPatternAnalysis } from './TemporalPatternAnalysis';
import {
  CognitivePattern,
  PatternConnection,
  CognitiveMetrics,
  InhibitionExcitationBalance as IEBalance,
  TemporalPattern,
  TemporalEvent
} from '../types/cognitive';

interface CognitivePatternDashboardProps {
  wsUrl?: string;
  className?: string;
}

export function CognitivePatternDashboard({ 
  wsUrl = 'ws://localhost:8080', 
  className = '' 
}: CognitivePatternDashboardProps) {
  const [activeTab, setActiveTab] = useState<'activation' | 'classification' | 'balance' | 'temporal'>('activation');
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Pattern data states
  const [patterns, setPatterns] = useState<CognitivePattern[]>(generateMockPatterns());
  const [connections, setConnections] = useState<PatternConnection[]>(generateMockConnections());
  const [metrics, setMetrics] = useState<CognitiveMetrics>(generateMockMetrics());
  const [balanceHistory, setBalanceHistory] = useState<IEBalance[]>(generateMockBalanceHistory());
  const [currentBalance, setCurrentBalance] = useState<IEBalance>(balanceHistory[balanceHistory.length - 1]);
  const [temporalPatterns, setTemporalPatterns] = useState<TemporalPattern[]>(generateMockTemporalPatterns());
  const [temporalEvents, setTemporalEvents] = useState<TemporalEvent[]>(generateMockTemporalEvents());

  // WebSocket connection
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: NodeJS.Timeout | null = null;

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          setIsConnected(true);
          console.log('Connected to cognitive pattern WebSocket');
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handlePatternUpdate(data);
            setLastUpdate(new Date());
          } catch (error) {
            console.error('Error parsing pattern data:', error);
          }
        };

        ws.onclose = () => {
          setIsConnected(false);
          console.log('Disconnected from cognitive pattern WebSocket');
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
        simulatePatternUpdates();
      }
    }, 2000);

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

  const handlePatternUpdate = (data: any) => {
    if (data.patterns) setPatterns(data.patterns);
    if (data.connections) setConnections(data.connections);
    if (data.metrics) setMetrics(data.metrics);
    if (data.balance) {
      setCurrentBalance(data.balance);
      setBalanceHistory(prev => [...prev.slice(-99), data.balance]);
    }
    if (data.temporalPatterns) setTemporalPatterns(data.temporalPatterns);
    if (data.temporalEvents) setTemporalEvents(prev => [...prev.slice(-500), ...data.temporalEvents]);
  };

  const simulatePatternUpdates = () => {
    // Simulate pattern activation changes
    setPatterns(prev => prev.map(pattern => ({
      ...pattern,
      activation: Math.max(0, Math.min(1, pattern.activation + (Math.random() - 0.5) * 0.1)),
      confidence: Math.max(0, Math.min(1, pattern.confidence + (Math.random() - 0.5) * 0.05))
    })));

    // Simulate balance changes
    const newBalance: IEBalance = {
      timestamp: Date.now(),
      excitation: {
        total: Math.random() * 100,
        byRegion: {
          prefrontal: Math.random() * 25,
          temporal: Math.random() * 25,
          parietal: Math.random() * 25,
          occipital: Math.random() * 25
        },
        patterns: patterns.filter(p => p.activation > 0.5).slice(0, 5).map(p => p.name)
      },
      inhibition: {
        total: Math.random() * 100,
        byRegion: {
          prefrontal: Math.random() * 25,
          temporal: Math.random() * 25,
          parietal: Math.random() * 25,
          occipital: Math.random() * 25
        },
        patterns: patterns.filter(p => p.activation < 0.3).slice(0, 5).map(p => p.name)
      },
      balance: 0,
      optimalRange: [-0.2, 0.2]
    };
    newBalance.balance = (newBalance.excitation.total - newBalance.inhibition.total) / 
                        (newBalance.excitation.total + newBalance.inhibition.total);
    
    setCurrentBalance(newBalance);
    setBalanceHistory(prev => [...prev.slice(-99), newBalance]);

    // Simulate temporal events
    const newEvent: TemporalEvent = {
      patternId: patterns[Math.floor(Math.random() * patterns.length)].name,
      timestamp: Date.now(),
      activation: Math.random(),
      context: ['context1', 'context2'].slice(0, Math.floor(Math.random() * 2) + 1)
    };
    setTemporalEvents(prev => [...prev.slice(-499), newEvent]);
  };

  const tabs = [
    { id: 'activation', label: '3D Activation', icon: '🧠' },
    { id: 'classification', label: 'Classification', icon: '📊' },
    { id: 'balance', label: 'I/E Balance', icon: '⚖️' },
    { id: 'temporal', label: 'Temporal', icon: '⏱️' }
  ];

  return (
    <div className={`min-h-screen bg-gray-950 text-gray-100 ${className}`}>
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold">Cognitive Pattern Visualization</h1>
            <p className="text-sm text-gray-400 mt-1">Real-time brain-inspired pattern analysis</p>
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
        {activeTab === 'activation' && (
          <PatternActivation3D 
            patterns={patterns} 
            connections={connections} 
          />
        )}
        
        {activeTab === 'classification' && (
          <PatternClassification 
            patterns={patterns} 
            metrics={metrics} 
          />
        )}
        
        {activeTab === 'balance' && (
          <InhibitionExcitationBalance 
            balanceData={balanceHistory} 
            currentBalance={currentBalance} 
          />
        )}
        
        {activeTab === 'temporal' && (
          <TemporalPatternAnalysis 
            patterns={temporalPatterns} 
            events={temporalEvents} 
          />
        )}

        {/* Quick Stats Bar */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-900 rounded p-4">
            <div className="text-sm text-gray-400">Active Patterns</div>
            <div className="text-2xl font-bold text-white">
              {patterns.filter(p => p.activation > 0.1).length}
            </div>
          </div>
          <div className="bg-gray-900 rounded p-4">
            <div className="text-sm text-gray-400">Avg Activation</div>
            <div className="text-2xl font-bold text-blue-400">
              {(patterns.reduce((sum, p) => sum + p.activation, 0) / patterns.length * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-gray-900 rounded p-4">
            <div className="text-sm text-gray-400">Balance Score</div>
            <div className={`text-2xl font-bold ${
              Math.abs(currentBalance.balance) < 0.2 ? 'text-green-400' : 
              Math.abs(currentBalance.balance) < 0.5 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {currentBalance.balance.toFixed(2)}
            </div>
          </div>
          <div className="bg-gray-900 rounded p-4">
            <div className="text-sm text-gray-400">Pattern Efficiency</div>
            <div className="text-2xl font-bold text-orange-400">
              {(metrics.performanceMetrics.resourceEfficiency * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Mock data generators
function generateMockPatterns(): CognitivePattern[] {
  const types = ['convergent', 'divergent', 'lateral', 'systems', 'critical', 'abstract', 'adaptive', 'chain_of_thought', 'tree_of_thoughts'];
  return Array.from({ length: 30 }, (_, i) => ({
    id: `pattern-${i}`,
    type: types[i % types.length] as any,
    name: `Pattern ${i}`,
    activation: Math.random(),
    confidence: 0.5 + Math.random() * 0.5,
    timestamp: Date.now() - Math.random() * 3600000,
    connections: [],
    metadata: {
      complexity: Math.random(),
      resourceUsage: {
        cpu: Math.random(),
        memory: Math.random(),
        duration: Math.random() * 1000
      },
      parameters: {},
      tags: ['tag1', 'tag2']
    }
  }));
}

function generateMockConnections(): PatternConnection[] {
  const patterns = generateMockPatterns();
  const connections: PatternConnection[] = [];
  
  for (let i = 0; i < 50; i++) {
    const source = patterns[Math.floor(Math.random() * patterns.length)];
    const target = patterns[Math.floor(Math.random() * patterns.length)];
    
    if (source.id !== target.id) {
      connections.push({
        sourceId: source.id,
        targetId: target.id,
        strength: Math.random(),
        type: Math.random() > 0.3 ? 'excitatory' : 'inhibitory'
      });
    }
  }
  
  return connections;
}

function generateMockMetrics(): CognitiveMetrics {
  const patterns = generateMockPatterns();
  const distribution: Record<string, number> = {};
  
  patterns.forEach(p => {
    distribution[p.type] = (distribution[p.type] || 0) + 1;
  });
  
  return {
    totalPatterns: patterns.length,
    activePatterns: patterns.filter(p => p.activation > 0.1).length,
    averageActivation: patterns.reduce((sum, p) => sum + p.activation, 0) / patterns.length,
    patternDistribution: distribution as any,
    performanceMetrics: {
      successRate: 0.85,
      averageLatency: 125,
      resourceEfficiency: 0.78
    }
  };
}

function generateMockBalanceHistory(): IEBalance[] {
  return Array.from({ length: 100 }, (_, i) => {
    const excitation = 50 + Math.sin(i * 0.1) * 30 + Math.random() * 10;
    const inhibition = 50 - Math.sin(i * 0.1) * 30 + Math.random() * 10;
    const balance = (excitation - inhibition) / (excitation + inhibition);
    
    return {
      timestamp: Date.now() - (100 - i) * 60000,
      excitation: {
        total: excitation,
        byRegion: {
          prefrontal: excitation * 0.3,
          temporal: excitation * 0.25,
          parietal: excitation * 0.25,
          occipital: excitation * 0.2
        },
        patterns: [`Pattern ${i % 10}`, `Pattern ${(i + 1) % 10}`]
      },
      inhibition: {
        total: inhibition,
        byRegion: {
          prefrontal: inhibition * 0.25,
          temporal: inhibition * 0.3,
          parietal: inhibition * 0.2,
          occipital: inhibition * 0.25
        },
        patterns: [`Pattern ${(i + 5) % 10}`, `Pattern ${(i + 6) % 10}`]
      },
      balance,
      optimalRange: [-0.2, 0.2]
    };
  });
}

function generateMockTemporalPatterns(): TemporalPattern[] {
  const patterns = generateMockPatterns();
  
  return Array.from({ length: 10 }, (_, i) => ({
    id: `temporal-${i}`,
    sequence: Array.from({ length: 3 + Math.floor(Math.random() * 5) }, (_, j) => ({
      patternId: patterns[Math.floor(Math.random() * patterns.length)].name,
      timestamp: Date.now() - (10 - j) * 1000,
      activation: Math.random(),
      context: ['context1', 'context2', 'context3'].slice(0, Math.floor(Math.random() * 3) + 1)
    })),
    frequency: Math.floor(Math.random() * 20) + 1,
    duration: Math.random() * 10000 + 5000,
    predictability: Math.random(),
    nextPredicted: Math.random() > 0.5 ? {
      patternId: patterns[Math.floor(Math.random() * patterns.length)].name,
      timestamp: Date.now() + Math.random() * 10000,
      activation: Math.random(),
      context: ['predicted']
    } : undefined
  }));
}

function generateMockTemporalEvents(): TemporalEvent[] {
  const patterns = generateMockPatterns();
  const events: TemporalEvent[] = [];
  
  for (let i = 0; i < 200; i++) {
    events.push({
      patternId: patterns[Math.floor(Math.random() * patterns.length)].name,
      timestamp: Date.now() - Math.random() * 3600000,
      activation: Math.random(),
      context: ['ctx1', 'ctx2', 'ctx3'].slice(0, Math.floor(Math.random() * 3) + 1)
    });
  }
  
  return events.sort((a, b) => a.timestamp - b.timestamp);
}