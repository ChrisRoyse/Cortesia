import React, { createContext, useContext, useEffect, useCallback } from 'react';
import { useAppDispatch, webSocketActions, dataActions } from '../stores';
import { WebSocketContextType, LLMKGData } from '../types';

// Use the same context name as WebSocketProvider to ensure compatibility
export const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface MockWebSocketProviderProps {
  children: React.ReactNode;
  enableMockData?: boolean;
}

// Mock data generator
const generateMockData = (): LLMKGData => {
  const timestamp = Date.now();
  const randomValue = () => Math.random() * 100;
  const randomPattern = () => ['alpha', 'beta', 'gamma', 'theta'][Math.floor(Math.random() * 4)];
  
  return {
    timestamp,
    cognitive: {
      patterns: [
        { id: '1', type: randomPattern(), strength: randomValue(), position: { x: randomValue(), y: randomValue(), z: randomValue() } },
        { id: '2', type: randomPattern(), strength: randomValue(), position: { x: randomValue(), y: randomValue(), z: randomValue() } },
      ],
      attention: {
        focus: randomValue(),
        distribution: Array.from({ length: 10 }, () => randomValue()),
      },
      inhibitory: {
        globalLevel: randomValue() / 100,
        localLevels: Array.from({ length: 5 }, () => randomValue() / 100),
      },
    },
    neural: {
      activity: Array.from({ length: 100 }, () => randomValue()),
      connectivity: Array.from({ length: 50 }, () => ({
        source: Math.floor(Math.random() * 100),
        target: Math.floor(Math.random() * 100),
        weight: Math.random(),
      })),
      spikes: Array.from({ length: 20 }, () => ({
        neuronId: Math.floor(Math.random() * 100),
        time: timestamp - Math.random() * 1000,
        amplitude: randomValue(),
      })),
    },
    memory: {
      workingMemory: {
        capacity: 7,
        usage: Math.floor(Math.random() * 7),
        items: [],
      },
      longTermMemory: {
        consolidationRate: randomValue() / 100,
        retrievalSpeed: randomValue(),
      },
    },
    performance: {
      cpu: randomValue(),
      memory: randomValue(),
      latency: Math.random() * 50,
      throughput: randomValue() * 10,
    },
  };
};

export const MockWebSocketProvider: React.FC<MockWebSocketProviderProps> = ({
  children,
  enableMockData = true,
}) => {
  const dispatch = useAppDispatch();
  
  useEffect(() => {
    // Simulate connection
    dispatch(webSocketActions.setConnectionState('connected'));
    
    if (enableMockData) {
      // Generate mock data periodically
      const interval = setInterval(() => {
        const mockData = generateMockData();
        dispatch(dataActions.setCurrentData(mockData));
      }, 1000);
      
      return () => {
        clearInterval(interval);
        dispatch(webSocketActions.setConnectionState('disconnected'));
      };
    }
    
    return () => {
      dispatch(webSocketActions.connectionStateChanged('disconnected'));
    };
  }, [dispatch, enableMockData]);
  
  const sendMessage = useCallback((message: any) => {
    console.log('Mock WebSocket: Message sent', message);
  }, []);
  
  const contextValue: WebSocketContextType = {
    isConnected: true,
    connectionState: 'connected',
    error: null,
    sendMessage,
    subscribe: () => () => {},
    lastMessage: null,
    latency: 10,
  };
  
  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};