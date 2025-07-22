/**
 * Complete Memory Visualization Integration Example
 * Shows how to use all memory visualization components together
 */

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import {
  MemoryVisualizationSystem,
  SDRPattern,
  MemoryOperation,
  StorageBlock,
  MemoryVisualizationSystemConfig
} from './src/memory';

interface MemoryDashboardProps {
  width?: number;
  height?: number;
  enableRealTime?: boolean;
}

export const MemoryDashboard: React.FC<MemoryDashboardProps> = ({
  width = 1200,
  height = 800,
  enableRealTime = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const systemRef = useRef<MemoryVisualizationSystem | null>(null);
  const animationRef = useRef<number>();
  
  const [metrics, setMetrics] = useState<any>(null);
  const [insights, setInsights] = useState<string[]>([]);
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    // Initialize the memory visualization system
    const config: MemoryVisualizationSystemConfig = {
      canvas: canvasRef.current,
      width,
      height,
      layout: 'grid',
      updateInterval: 1000,
      enableRealTimeUpdates: enableRealTime,
      enableCrossComponentAnalysis: true,
      maxHistorySize: 1000,
      
      // Component configurations
      sdrConfig: {
        maxPatterns: 500,
        cellSize: 2,
        gridDimensions: { rows: 32, cols: 64 }
      },
      
      operationConfig: {
        memorySize: 2 * 1024 * 1024 * 1024, // 2GB
        animationDuration: 1.5
      },
      
      efficiencyConfig: {
        maxBlocks: 1000,
        updateInterval: 500
      },
      
      analyticsConfig: {
        historySize: 500,
        alertThresholds: {
          fragmentation: 0.4,
          memoryUsage: 0.75,
          cacheHitRate: 0.6,
          compressionRatio: 1.2,
          queryLatency: 50
        }
      }
    };

    systemRef.current = new MemoryVisualizationSystem(config);

    // Set up event listeners
    canvasRef.current.addEventListener('sdr-pattern-selected', (event: any) => {
      setSelectedPattern(event.detail.patternId);
    });

    canvasRef.current.addEventListener('memory-insight-generated', (event: any) => {
      setInsights(prev => [...prev.slice(-4), event.detail.insight.title]);
    });

    // Start animation loop
    const animate = () => {
      if (systemRef.current) {
        systemRef.current.animate();
        
        // Update metrics periodically
        const newMetrics = systemRef.current.getPerformanceMetrics();
        setMetrics(newMetrics);
        
        // Get optimization recommendations
        const recommendations = systemRef.current.getOptimizationRecommendations();
        if (recommendations.length > 0) {
          setInsights(prev => [...prev.slice(-4), ...recommendations.slice(0, 1)]);
        }
      }
      
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    // Simulate real data
    if (enableRealTime) {
      simulateMemoryData();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (systemRef.current) {
        systemRef.current.dispose();
      }
    };
  }, [width, height, enableRealTime]);

  const simulateMemoryData = () => {
    if (!systemRef.current) return;

    // Simulate SDR patterns
    const simulateSDRPatterns = () => {
      for (let i = 0; i < 10; i++) {
        const activeBits = new Set<number>();
        const numActiveBits = 20 + Math.floor(Math.random() * 20);
        
        while (activeBits.size < numActiveBits) {
          activeBits.add(Math.floor(Math.random() * 2048));
        }

        const pattern: SDRPattern = {
          patternId: `pattern-${i}-${Date.now()}`,
          activeBits,
          totalBits: 2048,
          conceptName: `concept-${i}`,
          confidence: 0.7 + Math.random() * 0.3,
          usageCount: Math.floor(Math.random() * 100),
          timestamp: Date.now()
        };

        systemRef.current!.addSDRPattern(pattern);
      }
    };

    // Simulate memory operations
    const simulateMemoryOperations = () => {
      const operations: MemoryOperation['type'][] = ['read', 'write', 'update', 'delete'];
      
      for (let i = 0; i < 5; i++) {
        const operation: MemoryOperation = {
          id: `op-${Date.now()}-${i}`,
          type: operations[Math.floor(Math.random() * operations.length)],
          entityId: `entity-${Math.floor(Math.random() * 100)}`,
          address: Math.floor(Math.random() * 1000000000),
          size: 1024 * (1 + Math.floor(Math.random() * 100)),
          timestamp: Date.now(),
          success: Math.random() > 0.1
        };

        systemRef.current!.startMemoryOperation(operation);

        // Complete operation after delay
        setTimeout(() => {
          systemRef.current!.completeMemoryOperation(operation.id, operation.success);
        }, 1000 + Math.random() * 2000);
      }
    };

    // Simulate storage blocks
    const simulateStorageBlocks = () => {
      const blockTypes: StorageBlock['type'][] = ['data', 'index', 'cache', 'free'];
      
      for (let i = 0; i < 20; i++) {
        const block: StorageBlock = {
          id: `block-${Date.now()}-${i}`,
          address: Math.floor(Math.random() * 1000000000),
          size: 4096 * (1 + Math.floor(Math.random() * 10)),
          type: blockTypes[Math.floor(Math.random() * blockTypes.length)],
          compressionLevel: Math.random() * 0.5,
          accessFrequency: Math.random(),
          lastAccessed: Date.now() - Math.floor(Math.random() * 60000),
          fragmentLevel: Math.random() * 0.3,
          entityIds: [`entity-${Math.floor(Math.random() * 50)}`]
        };

        systemRef.current!.updateStorageBlock(block);
      }

      // Record I/O operations
      for (let i = 0; i < 5; i++) {
        systemRef.current!.recordIOOperation();
      }
    };

    // Initial simulation
    simulateSDRPatterns();
    simulateMemoryOperations();
    simulateStorageBlocks();

    // Periodic updates
    const intervals = [
      setInterval(simulateMemoryOperations, 3000),
      setInterval(simulateStorageBlocks, 5000),
      setInterval(() => {
        // Occasionally add new SDR patterns
        if (Math.random() > 0.7) {
          simulateSDRPatterns();
        }
      }, 7000)
    ];

    // Cleanup intervals after 2 minutes
    setTimeout(() => {
      intervals.forEach(clearInterval);
    }, 120000);
  };

  const handleExportMetrics = () => {
    if (metrics) {
      const dataStr = JSON.stringify(metrics, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `memory-metrics-${Date.now()}.json`;
      link.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Control Panel */}
      <div style={{
        padding: '16px',
        background: '#1a1a1a',
        color: '#ffffff',
        borderBottom: '1px solid #333',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h2 style={{ margin: '0 0 8px 0', fontSize: '18px' }}>LLMKG Memory Visualization</h2>
          <div style={{ fontSize: '12px', opacity: 0.7 }}>
            {selectedPattern ? `Selected Pattern: ${selectedPattern}` : 'No pattern selected'}
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <button 
            onClick={handleExportMetrics}
            style={{
              padding: '6px 12px',
              background: '#4a9eff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Export Metrics
          </button>
          
          <div style={{ fontSize: '12px' }}>
            {metrics && (
              <>
                Memory: {(metrics.system?.totalMemory / (1024 * 1024)).toFixed(1)}MB | 
                FPS: {metrics.sdr?.fps || 0} | 
                Operations: {metrics.operations?.operationsPerSecond || 0}/s
              </>
            )}
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div style={{ flex: 1, position: 'relative', background: '#000000' }}>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{ 
            width: '100%', 
            height: '100%',
            display: 'block'
          }}
        />
        
        {/* Overlay Info Panel */}
        <div style={{
          position: 'absolute',
          top: '16px',
          right: '16px',
          width: '300px',
          background: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '12px',
          borderRadius: '6px',
          fontSize: '12px',
          backdropFilter: 'blur(10px)'
        }}>
          <h4 style={{ margin: '0 0 12px 0', color: '#4a9eff' }}>System Insights</h4>
          {insights.length > 0 ? (
            <ul style={{ margin: 0, padding: '0 0 0 16px' }}>
              {insights.slice(-5).map((insight, index) => (
                <li key={index} style={{ marginBottom: '4px', opacity: 1 - (index * 0.1) }}>
                  {insight}
                </li>
              ))}
            </ul>
          ) : (
            <div style={{ opacity: 0.6 }}>No insights available yet...</div>
          )}
        </div>

        {/* Performance Metrics */}
        {metrics && (
          <div style={{
            position: 'absolute',
            bottom: '16px',
            left: '16px',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '12px',
            borderRadius: '6px',
            fontSize: '11px',
            backdropFilter: 'blur(10px)',
            minWidth: '200px'
          }}>
            <h4 style={{ margin: '0 0 8px 0', color: '#4a9eff' }}>Performance</h4>
            <div>SDR Patterns: {metrics.sdr?.patterns?.total || 0}</div>
            <div>Active Operations: {metrics.operations?.operationsPerSecond || 0}</div>
            <div>Storage Efficiency: {((metrics.efficiency?.compressionRatio || 1) * 100).toFixed(0)}%</div>
            <div>Memory Usage: {(metrics.system?.totalMemory / (1024 * 1024)).toFixed(1)}MB</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MemoryDashboard;