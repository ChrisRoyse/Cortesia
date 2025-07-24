import { useState, useEffect, useCallback, useRef } from 'react';
import {
  knowledgeGraphDataService,
  ComprehensiveKnowledgeGraphData,
  RealEntity,
  RealTriple,
  RealRelationship
} from '../services/KnowledgeGraphDataService';

export interface UseRealKnowledgeGraphOptions {
  autoRefresh?: boolean;
  refreshInterval?: number; // milliseconds
  enableWebSocket?: boolean;
}

export interface UseRealKnowledgeGraphResult {
  data: ComprehensiveKnowledgeGraphData | null;
  loading: boolean;
  error: string | null;
  refreshData: () => Promise<void>;
  searchEntities: (query: string) => Promise<RealEntity[]>;
  getEntityDetails: (entityId: string) => Promise<any>;
  clearCache: () => void;
  lastUpdated: number | null;
  connectionStatus: 'connected' | 'disconnected' | 'connecting' | 'error';
}

/**
 * Hook for managing real knowledge graph data with automatic refresh and real-time updates
 */
export function useRealKnowledgeGraph(
  options: UseRealKnowledgeGraphOptions = {}
): UseRealKnowledgeGraphResult {
  const {
    autoRefresh = true,
    refreshInterval = 30000, // 30 seconds
    enableWebSocket = true
  } = options;

  const [data, setData] = useState<ComprehensiveKnowledgeGraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting' | 'error'>('disconnected');

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const mountedRef = useRef(true);

  // Main data fetching function
  const refreshData = useCallback(async () => {
    if (!mountedRef.current) return;

    setLoading(true);
    setError(null);

    try {
      console.log('🔄 Refreshing knowledge graph data...');
      const comprehensiveData = await knowledgeGraphDataService.fetchComprehensiveData();
      
      if (mountedRef.current) {
        setData(comprehensiveData);
        setLastUpdated(Date.now());
        setConnectionStatus('connected');
        console.log('✅ Knowledge graph data updated:', {
          entities: comprehensiveData.entities.length,
          triples: comprehensiveData.triples.length,
          relationships: comprehensiveData.relationships.length,
          chunks: comprehensiveData.chunks.length
        });
      }
    } catch (err) {
      console.error('❌ Error refreshing knowledge graph data:', err);
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to refresh knowledge graph data');
        setConnectionStatus('error');
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  // Search entities function
  const searchEntities = useCallback(async (query: string): Promise<RealEntity[]> => {
    try {
      return await knowledgeGraphDataService.searchEntities(query);
    } catch (err) {
      console.error('Error searching entities:', err);
      return [];
    }
  }, []);

  // Get entity details function
  const getEntityDetails = useCallback(async (entityId: string) => {
    try {
      return await knowledgeGraphDataService.getEntityDetails(entityId);
    } catch (err) {
      console.error('Error getting entity details:', err);
      return null;
    }
  }, []);

  // Clear cache function
  const clearCache = useCallback(() => {
    knowledgeGraphDataService.clearCache();
    refreshData();
  }, [refreshData]);

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    if (!enableWebSocket || wsRef.current) return;

    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8081';
    console.log('🔌 Connecting to knowledge graph WebSocket:', wsUrl);
    
    setConnectionStatus('connecting');
    
    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('✅ Knowledge graph WebSocket connected');
        setConnectionStatus('connected');
        
        // Subscribe to knowledge graph updates
        if (wsRef.current) {
          wsRef.current.send(JSON.stringify({
            type: 'subscribe',
            topics: ['knowledge_graph_updates', 'entity_changes', 'triple_updates']
          }));
        }
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('📨 Knowledge graph update received:', message);

          // Handle different types of updates
          if (message.type === 'knowledge_graph_update' || 
              message.type === 'entity_update' || 
              message.type === 'triple_update') {
            
            // Refresh data when we receive updates
            // Debounce rapid updates
            setTimeout(() => {
              if (mountedRef.current) {
                refreshData();
              }
            }, 1000);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('❌ Knowledge graph WebSocket disconnected:', event.code, event.reason);
        setConnectionStatus('disconnected');
        wsRef.current = null;

        // Attempt to reconnect after 5 seconds
        if (mountedRef.current && enableWebSocket) {
          setTimeout(() => {
            if (mountedRef.current) {
              connectWebSocket();
            }
          }, 5000);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('🔥 Knowledge graph WebSocket error:', error);
        setConnectionStatus('error');
      };

    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setConnectionStatus('error');
    }
  }, [enableWebSocket, refreshData]);

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close(1000, 'Component unmounting');
      wsRef.current = null;
    }
  }, []);

  // Set up auto-refresh interval
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      intervalRef.current = setInterval(() => {
        if (mountedRef.current && !loading) {
          refreshData();
        }
      }, refreshInterval);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      };
    }
  }, [autoRefresh, refreshInterval, refreshData, loading]);

  // Initial data load and WebSocket connection
  useEffect(() => {
    refreshData();
    
    if (enableWebSocket) {
      connectWebSocket();
    }

    return () => {
      mountedRef.current = false;
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      
      disconnectWebSocket();
    };
  }, [refreshData, enableWebSocket, connectWebSocket, disconnectWebSocket]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  return {
    data,
    loading,
    error,
    refreshData,
    searchEntities,
    getEntityDetails,
    clearCache,
    lastUpdated,
    connectionStatus
  };
}

// Additional hook for entity-specific operations
export function useEntityOperations() {
  const [selectedEntity, setSelectedEntity] = useState<RealEntity | null>(null);
  const [entityDetails, setEntityDetails] = useState<any>(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  const selectEntity = useCallback(async (entity: RealEntity) => {
    setSelectedEntity(entity);
    setLoadingDetails(true);
    
    try {
      const details = await knowledgeGraphDataService.getEntityDetails(entity.id);
      setEntityDetails(details);
    } catch (err) {
      console.error('Error loading entity details:', err);
      setEntityDetails(null);
    } finally {
      setLoadingDetails(false);
    }
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedEntity(null);
    setEntityDetails(null);
  }, []);

  return {
    selectedEntity,
    entityDetails,
    loadingDetails,
    selectEntity,
    clearSelection
  };
}

// Hook for advanced search and filtering
export function useKnowledgeGraphSearch() {
  const [searchResults, setSearchResults] = useState<{
    entities: RealEntity[];
    triples: RealTriple[];
    relationships: RealRelationship[];
  }>({ entities: [], triples: [], relationships: [] });
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const performSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults({ entities: [], triples: [], relationships: [] });
      return;
    }

    setSearchLoading(true);
    setSearchQuery(query);

    try {
      // Perform parallel searches
      const [entities] = await Promise.all([
        knowledgeGraphDataService.searchEntities(query, 100)
      ]);

      // For triples and relationships, we'd need additional API endpoints
      // For now, we'll just return entities
      setSearchResults({
        entities,
        triples: [],
        relationships: []
      });

    } catch (err) {
      console.error('Error performing search:', err);
      setSearchResults({ entities: [], triples: [], relationships: [] });
    } finally {
      setSearchLoading(false);
    }
  }, []);

  const clearSearch = useCallback(() => {
    setSearchQuery('');
    setSearchResults({ entities: [], triples: [], relationships: [] });
  }, []);

  return {
    searchResults,
    searchLoading,
    searchQuery,
    performSearch,
    clearSearch
  };
}