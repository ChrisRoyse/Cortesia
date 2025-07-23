import { useState, useEffect, useCallback, useRef } from 'react';
import { TestWebSocketService, TestStreamingEvent } from '../services/TestWebSocketService';
import { TestExecution } from '../services/TestExecutionTracker';

export interface TestStreamingState {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  activeExecutions: Map<string, TestExecution>;
  recentEvents: TestStreamingEvent[];
  executionLogs: Map<string, TestStreamingEvent[]>;
}

export interface TestStreamingActions {
  connect: () => Promise<void>;
  disconnect: () => void;
  subscribeToExecution: (executionId: string) => void;
  unsubscribeFromExecution: (executionId: string) => void;
  subscribeToAllExecutions: () => void;
  startTestExecution: (suiteId: string, options?: any) => void;
  cancelTestExecution: (executionId: string) => void;
  clearEvents: () => void;
  getExecutionEvents: (executionId: string) => TestStreamingEvent[];
}

export const useTestStreaming = (
  webSocketService?: TestWebSocketService,
  options: {
    autoConnect?: boolean;
    maxRecentEvents?: number;
    maxExecutionLogs?: number;
  } = {}
): [TestStreamingState, TestStreamingActions] => {
  const { 
    autoConnect = true, 
    maxRecentEvents = 100, 
    maxExecutionLogs = 50 
  } = options;
  
  const serviceRef = useRef<TestWebSocketService>(
    webSocketService || TestWebSocketService.createMockService()
  );
  
  const [state, setState] = useState<TestStreamingState>({
    connected: false,
    connecting: false,
    error: null,
    activeExecutions: new Map(),
    recentEvents: [],
    executionLogs: new Map()
  });

  // Handle WebSocket events
  useEffect(() => {
    const service = serviceRef.current;

    const handleConnected = () => {
      setState(prev => ({
        ...prev,
        connected: true,
        connecting: false,
        error: null
      }));
    };

    const handleDisconnected = () => {
      setState(prev => ({
        ...prev,
        connected: false,
        connecting: false
      }));
    };

    const handleError = (error: any) => {
      setState(prev => ({
        ...prev,
        error: error.message || 'WebSocket connection error',
        connecting: false
      }));
    };

    const handleStreamingEvent = (event: TestStreamingEvent) => {
      setState(prev => {
        const newState = { ...prev };
        
        // Update recent events
        newState.recentEvents = [event, ...prev.recentEvents].slice(0, maxRecentEvents);
        
        // Update execution logs
        const executionLogs = new Map(prev.executionLogs);
        const executionEvents = executionLogs.get(event.executionId) || [];
        executionLogs.set(event.executionId, [event, ...executionEvents].slice(0, maxExecutionLogs));
        newState.executionLogs = executionLogs;
        
        // Update active executions
        const activeExecutions = new Map(prev.activeExecutions);
        
        switch (event.event) {
          case 'started':
            activeExecutions.set(event.executionId, {
              id: event.executionId,
              startTime: event.timestamp,
              status: 'running',
              testPattern: event.data.suiteId,
              category: event.data.suiteName,
              progress: {
                current: 0,
                total: event.data.totalTests || 0
              },
              logs: []
            });
            break;
            
          case 'progress':
            const progressExecution = activeExecutions.get(event.executionId);
            if (progressExecution) {
              progressExecution.progress = {
                current: event.data.current || progressExecution.progress.current,
                total: event.data.total || progressExecution.progress.total,
                currentTest: event.data.currentTest
              };
            }
            break;
            
          case 'completed':
            const completedExecution = activeExecutions.get(event.executionId);
            if (completedExecution) {
              completedExecution.status = 'completed';
              completedExecution.endTime = event.timestamp;
              completedExecution.summary = {
                passed: event.data.passed || 0,
                failed: event.data.failed || 0,
                ignored: event.data.ignored || 0,
                total: (event.data.passed || 0) + (event.data.failed || 0) + (event.data.ignored || 0),
                executionTime: event.data.executionTime || 0,
                results: event.data.results || []
              };
              // Remove from active executions after a delay
              setTimeout(() => {
                setState(prev => {
                  const newActiveExecutions = new Map(prev.activeExecutions);
                  newActiveExecutions.delete(event.executionId);
                  return { ...prev, activeExecutions: newActiveExecutions };
                });
              }, 5000);
            }
            break;
            
          case 'failed':
            const failedExecution = activeExecutions.get(event.executionId);
            if (failedExecution) {
              failedExecution.status = 'failed';
              failedExecution.endTime = event.timestamp;
            }
            break;
            
          case 'cancelled':
            const cancelledExecution = activeExecutions.get(event.executionId);
            if (cancelledExecution) {
              cancelledExecution.status = 'cancelled';
              cancelledExecution.endTime = event.timestamp;
            }
            break;
            
          case 'log':
            const logExecution = activeExecutions.get(event.executionId);
            if (logExecution) {
              logExecution.logs.push({
                timestamp: event.timestamp,
                level: event.data.level || 'info',
                message: event.data.message || '',
                testName: event.data.testName
              });
              // Keep only recent logs
              if (logExecution.logs.length > 50) {
                logExecution.logs = logExecution.logs.slice(-50);
              }
            }
            break;
        }
        
        newState.activeExecutions = activeExecutions;
        return newState;
      });
    };

    // Set up event listeners
    service.on('connected', handleConnected);
    service.on('disconnected', handleDisconnected);
    service.on('error', handleError);
    service.on('streaming_event', handleStreamingEvent);

    // Auto-connect if requested
    if (autoConnect && !service.isConnected()) {
      setState(prev => ({ ...prev, connecting: true }));
      service.connect().catch(error => {
        console.error('Failed to auto-connect to test streaming:', error);
      });
    }

    return () => {
      service.off('connected', handleConnected);
      service.off('disconnected', handleDisconnected);
      service.off('error', handleError);
      service.off('streaming_event', handleStreamingEvent);
    };
  }, [autoConnect, maxRecentEvents, maxExecutionLogs]);

  // Actions
  const connect = useCallback(async () => {
    setState(prev => ({ ...prev, connecting: true, error: null }));
    try {
      await serviceRef.current.connect();
    } catch (error) {
      setState(prev => ({
        ...prev,
        connecting: false,
        error: error instanceof Error ? error.message : 'Connection failed'
      }));
    }
  }, []);

  const disconnect = useCallback(() => {
    serviceRef.current.disconnect();
    setState(prev => ({
      ...prev,
      connected: false,
      connecting: false,
      activeExecutions: new Map()
    }));
  }, []);

  const subscribeToExecution = useCallback((executionId: string) => {
    serviceRef.current.subscribeToExecution(executionId);
  }, []);

  const unsubscribeFromExecution = useCallback((executionId: string) => {
    serviceRef.current.unsubscribeFromExecution(executionId);
  }, []);

  const subscribeToAllExecutions = useCallback(() => {
    serviceRef.current.subscribeToAllExecutions();
  }, []);

  const startTestExecution = useCallback((suiteId: string, options?: any) => {
    serviceRef.current.startTestExecution(suiteId, options);
  }, []);

  const cancelTestExecution = useCallback((executionId: string) => {
    serviceRef.current.cancelTestExecution(executionId);
  }, []);

  const clearEvents = useCallback(() => {
    setState(prev => ({
      ...prev,
      recentEvents: [],
      executionLogs: new Map()
    }));
  }, []);

  const getExecutionEvents = useCallback((executionId: string): TestStreamingEvent[] => {
    return state.executionLogs.get(executionId) || [];
  }, [state.executionLogs]);

  const actions: TestStreamingActions = {
    connect,
    disconnect,
    subscribeToExecution,
    unsubscribeFromExecution,
    subscribeToAllExecutions,
    startTestExecution,
    cancelTestExecution,
    clearEvents,
    getExecutionEvents
  };

  return [state, actions];
};