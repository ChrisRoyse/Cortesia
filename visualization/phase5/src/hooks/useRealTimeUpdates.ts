import { useState, useEffect, useMemo } from 'react';
import { Observable, Subject, BehaviorSubject, fromEvent, merge, interval } from 'rxjs';
import { map, filter, throttleTime, distinctUntilChanged, catchError } from 'rxjs/operators';
import { 
  ArchitectureUpdate, 
  ComponentUpdate,
  TelemetryData,
  WebSocketConnection,
  ComponentStatus
} from '../types';

export interface RealTimeConfig {
  updateInterval: number;
  maxBufferSize: number;
  reconnectInterval: number;
  enableTelemetry: boolean;
  enableWebSocket: boolean;
}

const defaultConfig: RealTimeConfig = {
  updateInterval: 1000,
  maxBufferSize: 100,
  reconnectInterval: 5000,
  enableTelemetry: true,
  enableWebSocket: true,
};

export function useRealTimeUpdates(
  enabled: boolean,
  websocketConnection?: WebSocketConnection,
  telemetryStream?: Observable<TelemetryData>,
  config: Partial<RealTimeConfig> = {}
) {
  const finalConfig = { ...defaultConfig, ...config };
  
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now());
  const [updateCount, setUpdateCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  
  // Buffer for storing recent updates
  const [updateBuffer, setUpdateBuffer] = useState<ArchitectureUpdate[]>([]);
  
  // Subject for manual updates
  const [manualUpdateSubject] = useState(() => new Subject<ArchitectureUpdate>());
  
  // Main real-time updates observable
  const realTimeUpdates = useMemo(() => {
    if (!enabled) return null;

    const streams: Observable<ArchitectureUpdate>[] = [];

    // WebSocket stream
    if (finalConfig.enableWebSocket && websocketConnection) {
      const wsStream = new Observable<ArchitectureUpdate>(subscriber => {
        const handleUpdate = (data: any) => {
          try {
            const update = transformWebSocketData(data);
            if (update) {
              subscriber.next(update);
            }
          } catch (error) {
            console.error('WebSocket data transformation error:', error);
            subscriber.error(error);
          }
        };

        // Subscribe to WebSocket updates
        websocketConnection.subscribe('architecture-updates', handleUpdate);
        websocketConnection.subscribe('component-updates', handleUpdate);
        websocketConnection.subscribe('system-events', handleUpdate);

        return () => {
          websocketConnection.unsubscribe('architecture-updates');
          websocketConnection.unsubscribe('component-updates');
          websocketConnection.unsubscribe('system-events');
        };
      }).pipe(
        catchError(error => {
          console.error('WebSocket stream error:', error);
          setErrorCount(prev => prev + 1);
          return [];
        })
      );

      streams.push(wsStream);
    }

    // Telemetry stream
    if (finalConfig.enableTelemetry && telemetryStream) {
      const telemetryUpdateStream = telemetryStream.pipe(
        map(data => transformTelemetryData(data)),
        filter(update => update !== null),
        map(update => update!),
        catchError(error => {
          console.error('Telemetry stream error:', error);
          setErrorCount(prev => prev + 1);
          return [];
        })
      );

      streams.push(telemetryUpdateStream);
    }

    // Manual updates stream
    streams.push(manualUpdateSubject.asObservable());

    // Heartbeat stream for connection monitoring
    const heartbeatStream = interval(finalConfig.updateInterval).pipe(
      map(() => createHeartbeatUpdate()),
      filter(update => update !== null),
      map(update => update!)
    );

    streams.push(heartbeatStream);

    if (streams.length === 0) return null;

    // Merge all streams
    const combinedStream = merge(...streams).pipe(
      // Throttle updates to prevent overwhelming
      throttleTime(100),
      // Remove duplicate updates
      distinctUntilChanged((a, b) => 
        a.timestamp === b.timestamp && 
        JSON.stringify(a.changes) === JSON.stringify(b.changes)
      )
    );

    return combinedStream;
  }, [
    enabled,
    websocketConnection,
    telemetryStream,
    finalConfig,
    manualUpdateSubject
  ]);

  // Monitor connection status
  useEffect(() => {
    if (!enabled || !websocketConnection) {
      setConnectionStatus('disconnected');
      setIsConnected(false);
      return;
    }

    const checkConnection = () => {
      const connected = websocketConnection.isConnected();
      setIsConnected(connected);
      setConnectionStatus(connected ? 'connected' : 'disconnected');
    };

    // Check connection immediately
    checkConnection();

    // Set up periodic connection checks
    const interval = setInterval(checkConnection, 1000);

    return () => clearInterval(interval);
  }, [enabled, websocketConnection]);

  // Subscribe to real-time updates and manage buffer
  useEffect(() => {
    if (!realTimeUpdates) return;

    const subscription = realTimeUpdates.subscribe({
      next: (update) => {
        setLastUpdate(Date.now());
        setUpdateCount(prev => prev + 1);
        
        // Add to buffer
        setUpdateBuffer(prev => {
          const newBuffer = [...prev, update];
          // Keep only recent updates within buffer size
          return newBuffer.slice(-finalConfig.maxBufferSize);
        });
      },
      error: (error) => {
        console.error('Real-time updates error:', error);
        setErrorCount(prev => prev + 1);
      }
    });

    return () => subscription.unsubscribe();
  }, [realTimeUpdates, finalConfig.maxBufferSize]);

  // Connection recovery
  useEffect(() => {
    if (!enabled || connectionStatus !== 'disconnected' || !websocketConnection) return;

    const attemptReconnect = () => {
      setConnectionStatus('connecting');
      // In a real implementation, you would call websocketConnection.reconnect()
      setTimeout(() => {
        if (websocketConnection.isConnected()) {
          setConnectionStatus('connected');
          setIsConnected(true);
        } else {
          setConnectionStatus('disconnected');
          setIsConnected(false);
        }
      }, 1000);
    };

    const reconnectTimer = setTimeout(attemptReconnect, finalConfig.reconnectInterval);
    return () => clearTimeout(reconnectTimer);
  }, [enabled, connectionStatus, websocketConnection, finalConfig.reconnectInterval]);

  // Manual update functions
  const pushUpdate = (update: ArchitectureUpdate) => {
    manualUpdateSubject.next(update);
  };

  const pushComponentUpdate = (componentId: string, changes: ComponentUpdate['changes']) => {
    const update: ArchitectureUpdate = {
      type: 'component-update',
      timestamp: Date.now(),
      changes: [{ componentId, changes }]
    };
    pushUpdate(update);
  };

  // Statistics
  const statistics = useMemo(() => {
    const now = Date.now();
    const recentUpdates = updateBuffer.filter(update => 
      now - update.timestamp < 60000 // Last minute
    );

    return {
      totalUpdates: updateCount,
      recentUpdates: recentUpdates.length,
      errorCount,
      averageUpdateRate: recentUpdates.length / 60, // Updates per second
      lastUpdateAge: now - lastUpdate,
      bufferSize: updateBuffer.length,
      isHealthy: isConnected && (now - lastUpdate) < 30000, // Healthy if last update within 30s
    };
  }, [updateCount, errorCount, lastUpdate, updateBuffer, isConnected]);

  // Get recent updates by type
  const getUpdatesByType = (type: ArchitectureUpdate['type'], limit: number = 10) => {
    return updateBuffer
      .filter(update => update.type === type)
      .slice(-limit)
      .reverse(); // Most recent first
  };

  // Get updates for specific component
  const getComponentUpdates = (componentId: string, limit: number = 10) => {
    return updateBuffer
      .filter(update => 
        update.changes.some(change => change.componentId === componentId)
      )
      .slice(-limit)
      .reverse(); // Most recent first
  };

  // Clear update buffer
  const clearBuffer = () => {
    setUpdateBuffer([]);
  };

  return {
    realTimeUpdates,
    isConnected,
    connectionStatus,
    statistics,
    updateBuffer,
    pushUpdate,
    pushComponentUpdate,
    getUpdatesByType,
    getComponentUpdates,
    clearBuffer,
  };
}

// Transform WebSocket data to ArchitectureUpdate
function transformWebSocketData(data: any): ArchitectureUpdate | null {
  try {
    // Handle different WebSocket message formats
    if (data.type === 'component-update') {
      return {
        type: 'component-update',
        timestamp: data.timestamp || Date.now(),
        changes: [{
          componentId: data.componentId,
          changes: data.changes
        }]
      };
    }

    if (data.type === 'system-event') {
      // Transform system events to component updates
      return {
        type: 'component-update',
        timestamp: data.timestamp || Date.now(),
        changes: data.affectedComponents?.map((componentId: string) => ({
          componentId,
          changes: { status: data.newStatus }
        })) || []
      };
    }

    if (data.type === 'architecture-change') {
      return {
        type: 'layout-change',
        timestamp: data.timestamp || Date.now(),
        changes: data.changes || []
      };
    }

    return null;
  } catch (error) {
    console.error('Error transforming WebSocket data:', error);
    return null;
  }
}

// Transform telemetry data to ArchitectureUpdate
function transformTelemetryData(data: TelemetryData): ArchitectureUpdate | null {
  try {
    const changes: ComponentUpdate[] = [];

    // Extract component updates from telemetry metrics
    if (data.metrics) {
      for (const [source, metrics] of Object.entries(data.metrics)) {
        if (typeof metrics === 'object' && metrics !== null) {
          changes.push({
            componentId: source,
            changes: {
              metrics: {
                cpu: { current: metrics.cpu || 0, average: metrics.cpu || 0, peak: metrics.cpu || 0 },
                memory: { current: metrics.memory || 0, average: metrics.memory || 0, peak: metrics.memory || 0 },
                throughput: { current: metrics.throughput || 0, average: metrics.throughput || 0, peak: metrics.throughput || 0 },
                latency: { current: metrics.latency || 0, average: metrics.latency || 0, peak: metrics.latency || 0 },
                errorRate: { current: metrics.errorRate || 0, average: metrics.errorRate || 0, peak: metrics.errorRate || 0 },
                lastUpdated: data.timestamp
              }
            }
          });
        }
      }
    }

    // Process telemetry events
    if (data.events && Array.isArray(data.events)) {
      for (const event of data.events) {
        if (event.type === 'status-change' && event.data?.componentId) {
          changes.push({
            componentId: event.data.componentId,
            changes: {
              status: event.data.newStatus as ComponentStatus
            }
          });
        }
      }
    }

    if (changes.length === 0) return null;

    return {
      type: 'component-update',
      timestamp: data.timestamp,
      changes
    };
  } catch (error) {
    console.error('Error transforming telemetry data:', error);
    return null;
  }
}

// Create heartbeat update for connection monitoring
function createHeartbeatUpdate(): ArchitectureUpdate | null {
  // Only create heartbeat updates periodically
  const now = Date.now();
  const shouldSendHeartbeat = Math.random() < 0.1; // 10% chance per interval

  if (!shouldSendHeartbeat) return null;

  return {
    type: 'component-update',
    timestamp: now,
    changes: [] // Empty changes for heartbeat
  };
}