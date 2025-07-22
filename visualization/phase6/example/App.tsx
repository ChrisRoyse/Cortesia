import React, { useState, useEffect } from 'react';
import { ConfigProvider, theme } from 'antd';
import { PerformanceDashboard } from '../src';
import type { PerformanceMetrics } from '../src/types';

// Mock WebSocket for demo
class MockWebSocket {
  private handlers: Map<string, (event: any) => void> = new Map();
  private interval: NodeJS.Timeout | null = null;
  
  constructor(url: string) {
    console.log('Mock WebSocket connected to:', url);
    setTimeout(() => this.simulateOpen(), 100);
  }

  set onopen(handler: () => void) {
    this.handlers.set('open', handler);
  }

  set onmessage(handler: (event: any) => void) {
    this.handlers.set('message', handler);
  }

  set onerror(handler: (error: any) => void) {
    this.handlers.set('error', handler);
  }

  set onclose(handler: () => void) {
    this.handlers.set('close', handler);
  }

  private simulateOpen() {
    const openHandler = this.handlers.get('open');
    if (openHandler) openHandler();
    this.startDataGeneration();
  }

  private startDataGeneration() {
    this.interval = setInterval(() => {
      const metrics = this.generateMockMetrics();
      const messageHandler = this.handlers.get('message');
      if (messageHandler) {
        messageHandler({
          data: JSON.stringify({
            type: 'performance_metrics',
            metrics
          })
        });
      }
    }, 1000);
  }

  private generateMockMetrics(): PerformanceMetrics {
    const timestamp = Date.now();
    const baseLatency = 30 + Math.sin(timestamp / 10000) * 20;
    
    return {
      timestamp,
      cognitive: {
        subcortical: {
          activationRate: 0.7 + Math.random() * 0.2,
          inhibitionRate: 0.3 + Math.random() * 0.1,
          processingLatency: baseLatency + Math.random() * 10,
          throughput: 800 + Math.random() * 200,
          errorCount: Math.random() > 0.95 ? 1 : 0,
          hebbianLearningRate: 0.5 + Math.random() * 0.3,
          attentionFocus: 0.6 + Math.random() * 0.3,
          cognitiveLoad: 0.4 + Math.random() * 0.4
        },
        cortical: {
          activationRate: 0.8 + Math.random() * 0.15,
          inhibitionRate: 0.2 + Math.random() * 0.1,
          processingLatency: baseLatency + 15 + Math.random() * 15,
          throughput: 600 + Math.random() * 300,
          errorCount: Math.random() > 0.98 ? 1 : 0,
          hebbianLearningRate: 0.6 + Math.random() * 0.2,
          attentionFocus: 0.7 + Math.random() * 0.2,
          cognitiveLoad: 0.5 + Math.random() * 0.3
        },
        thalamic: {
          activationRate: 0.6 + Math.random() * 0.3,
          inhibitionRate: 0.4 + Math.random() * 0.2,
          processingLatency: baseLatency - 10 + Math.random() * 20,
          throughput: 1000 + Math.random() * 500,
          errorCount: Math.random() > 0.99 ? 1 : 0,
          hebbianLearningRate: 0.4 + Math.random() * 0.4,
          attentionFocus: 0.8 + Math.random() * 0.15,
          cognitiveLoad: 0.3 + Math.random() * 0.5
        }
      },
      sdr: {
        creationRate: 100 + Math.random() * 50,
        averageSparsity: 0.02 + Math.random() * 0.03,
        overlapRatio: 0.1 + Math.random() * 0.2,
        memoryUsage: 500000000 + Math.random() * 500000000,
        compressionRatio: 8 + Math.random() * 4,
        semanticAccuracy: 0.85 + Math.random() * 0.1,
        storageEfficiency: 0.7 + Math.random() * 0.2
      },
      mcp: {
        messageRate: 500 + Math.random() * 500,
        averageLatency: 20 + Math.random() * 30,
        errorRate: Math.random() * 0.02,
        queueLength: Math.floor(Math.random() * 150),
        protocolOverhead: 5 + Math.random() * 10,
        throughput: 1000 + Math.random() * 1000,
        reliability: 0.95 + Math.random() * 0.04
      },
      system: {
        cpuUsage: 40 + Math.sin(timestamp / 5000) * 30 + Math.random() * 10,
        memoryUsage: 50 + Math.sin(timestamp / 7000) * 20 + Math.random() * 10,
        diskIO: 20 + Math.random() * 40,
        networkIO: 30 + Math.random() * 50,
        gpuUsage: 10 + Math.random() * 30,
        cacheHitRate: 0.8 + Math.random() * 0.15
      }
    };
  }

  send(data: string) {
    console.log('Mock WebSocket send:', data);
  }

  close() {
    if (this.interval) {
      clearInterval(this.interval);
    }
    const closeHandler = this.handlers.get('close');
    if (closeHandler) closeHandler();
  }

  get readyState() {
    return 1; // OPEN
  }

  static OPEN = 1;
}

// Replace global WebSocket with mock for demo
(window as any).WebSocket = MockWebSocket;

export default function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const { defaultAlgorithm, darkAlgorithm } = theme;

  return (
    <ConfigProvider
      theme={{
        algorithm: isDarkMode ? darkAlgorithm : defaultAlgorithm,
        token: {
          colorPrimary: '#1890ff',
        },
      }}
    >
      <div style={{ 
        minHeight: '100vh',
        backgroundColor: isDarkMode ? '#141414' : '#f0f2f5',
        padding: 24
      }}>
        <div style={{ 
          maxWidth: 1600, 
          margin: '0 auto',
          backgroundColor: isDarkMode ? '#1f1f1f' : '#fff',
          borderRadius: 8,
          padding: 24,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: 24
          }}>
            <h1 style={{ margin: 0, color: isDarkMode ? '#fff' : '#000' }}>
              LLMKG Performance Monitoring - Phase 6
            </h1>
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              style={{
                padding: '8px 16px',
                backgroundColor: isDarkMode ? '#434343' : '#f0f0f0',
                color: isDarkMode ? '#fff' : '#000',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer'
              }}
            >
              {isDarkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
            </button>
          </div>

          <PerformanceDashboard
            websocketUrl="ws://localhost:8080/performance"
            theme={isDarkMode ? 'dark' : 'light'}
            showAlerts={true}
            showOptimizations={true}
            refreshInterval={1000}
          />
        </div>
      </div>
    </ConfigProvider>
  );
}