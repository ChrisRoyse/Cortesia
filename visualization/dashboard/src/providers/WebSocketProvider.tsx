import React, { createContext, useContext, useEffect, useRef, useCallback, useMemo } from 'react';
import { useAppDispatch, useAppSelector, webSocketActions, dataActions } from '../stores';
import { WebSocketContextType, WebSocketMessage, LLMKGData } from '../types';

const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  url: string;
  children: React.ReactNode;
  reconnectDelay?: number;
  heartbeatInterval?: number;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  url,
  children,
  reconnectDelay = 3000,
  heartbeatInterval = 30000,
}) => {
  const dispatch = useAppDispatch();
  const { 
    isConnected, 
    connectionState, 
    error, 
    reconnectAttempts, 
    maxReconnectAttempts 
  } = useAppSelector(state => state.webSocket);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    clearTimeouts();
    heartbeatTimeoutRef.current = setTimeout(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const pingMessage: WebSocketMessage = {
          type: 'ping',
          timestamp: Date.now(),
        };
        wsRef.current.send(JSON.stringify(pingMessage));
        startHeartbeat();
      }
    }, heartbeatInterval);
  }, [heartbeatInterval, clearTimeouts]);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      dispatch(webSocketActions.setLastMessage(message));

      switch (message.type) {
        case 'data':
          if (message.data && isValidLLMKGData(message.data)) {
            dispatch(dataActions.setCurrentData(message.data as LLMKGData));
            dispatch(dataActions.setError({ hasError: false, error: null }));
          }
          break;

        case 'error':
          dispatch(webSocketActions.setError(message.error || 'Unknown WebSocket error'));
          dispatch(dataActions.setError({ 
            hasError: true, 
            error: new Error(message.error || 'WebSocket error') 
          }));
          break;

        case 'pong':
          // Heartbeat response received
          break;

        default:
          console.log('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      dispatch(webSocketActions.setError('Failed to parse message'));
    }
  }, [dispatch]);

  const handleOpen = useCallback(() => {
    dispatch(webSocketActions.setConnectionState('connected'));
    dispatch(webSocketActions.setError(null));
    dispatch(webSocketActions.resetReconnectAttempts());
    dispatch(dataActions.setError({ hasError: false, error: null }));
    startHeartbeat();
  }, [dispatch, startHeartbeat]);

  const handleClose = useCallback((event: CloseEvent) => {
    dispatch(webSocketActions.setConnectionState('disconnected'));
    clearTimeouts();

    if (mountedRef.current && !event.wasClean && reconnectAttempts < maxReconnectAttempts) {
      const delay = Math.min(reconnectDelay * Math.pow(2, reconnectAttempts), 30000);
      reconnectTimeoutRef.current = setTimeout(() => {
        if (mountedRef.current) {
          dispatch(webSocketActions.incrementReconnectAttempts());
          connect();
        }
      }, delay);
    } else if (reconnectAttempts >= maxReconnectAttempts) {
      dispatch(webSocketActions.setError('Max reconnection attempts reached'));
    }
  }, [dispatch, reconnectDelay, reconnectAttempts, maxReconnectAttempts, clearTimeouts]);

  const handleError = useCallback((event: Event) => {
    dispatch(webSocketActions.setConnectionState('error'));
    dispatch(webSocketActions.setError('WebSocket connection error'));
    console.error('WebSocket error:', event);
  }, [dispatch]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      dispatch(webSocketActions.setConnectionState('connecting'));
      dispatch(webSocketActions.setError(null));

      wsRef.current = new WebSocket(url);
      wsRef.current.onopen = handleOpen;
      wsRef.current.onmessage = handleMessage;
      wsRef.current.onclose = handleClose;
      wsRef.current.onerror = handleError;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      dispatch(webSocketActions.setError('Failed to create connection'));
    }
  }, [url, dispatch, handleOpen, handleMessage, handleClose, handleError]);

  const disconnect = useCallback(() => {
    mountedRef.current = false;
    clearTimeouts();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
    
    dispatch(webSocketActions.setConnectionState('disconnected'));
  }, [dispatch, clearTimeouts]);

  const send = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          ...message,
          timestamp: Date.now(),
        }));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        dispatch(webSocketActions.setError('Failed to send message'));
      }
    } else {
      console.warn('WebSocket is not connected. Cannot send message:', message);
    }
  }, [dispatch]);

  const subscribe = useCallback((topics: string[]) => {
    const message: WebSocketMessage = {
      type: 'subscribe',
      topics,
      timestamp: Date.now(),
    };
    send(message);
    dispatch(dataActions.setSubscriptions(topics));
  }, [send, dispatch]);

  const unsubscribe = useCallback((topics: string[]) => {
    const message: WebSocketMessage = {
      type: 'unsubscribe',
      topics,
      timestamp: Date.now(),
    };
    send(message);
    
    // Update subscriptions by removing the unsubscribed topics
    const currentSubscriptions = useAppSelector(state => state.data.subscriptions);
    const newSubscriptions = currentSubscriptions.filter(topic => !topics.includes(topic));
    dispatch(dataActions.setSubscriptions(newSubscriptions));
  }, [send, dispatch]);

  // Initialize connection on mount
  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [connect, disconnect]);

  // Get current data from store
  const currentData = useAppSelector(state => state.data.current);
  const lastMessage = useAppSelector(state => state.webSocket.lastMessage);

  const contextValue = useMemo<WebSocketContextType>(() => ({
    isConnected,
    connectionState,
    data: currentData,
    lastMessage,
    send,
    subscribe,
    unsubscribe,
    error,
  }), [isConnected, connectionState, currentData, lastMessage, send, subscribe, unsubscribe, error]);

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Hook to use WebSocket context
export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

// Utility function to validate LLMKG data structure
function isValidLLMKGData(data: any): data is LLMKGData {
  return (
    data &&
    typeof data === 'object' &&
    data.cognitive &&
    data.neural &&
    data.knowledgeGraph &&
    data.memory &&
    typeof data.timestamp === 'number'
  );
}

// Custom hook for subscribing to specific topics
export const useWebSocketSubscription = (topics: string[], autoSubscribe = true) => {
  const { subscribe, unsubscribe, isConnected } = useWebSocket();

  useEffect(() => {
    if (autoSubscribe && isConnected && topics.length > 0) {
      subscribe(topics);
    }

    return () => {
      if (autoSubscribe && topics.length > 0) {
        unsubscribe(topics);
      }
    };
  }, [topics, autoSubscribe, isConnected, subscribe, unsubscribe]);
};

// Custom hook for real-time data with filtering
export const useRealtimeData = <T = LLMKGData>(
  selector?: (data: LLMKGData) => T,
  dependencies: React.DependencyList = []
) => {
  const { data, isConnected } = useWebSocket();
  
  return useMemo(() => {
    if (!data || !isConnected) return null;
    return selector ? selector(data) : (data as unknown as T);
  }, [data, isConnected, selector, ...dependencies]);
};