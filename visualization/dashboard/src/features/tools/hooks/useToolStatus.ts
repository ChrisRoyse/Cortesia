import { useState, useEffect, useCallback, useRef } from 'react';
import { useAppSelector, useAppDispatch } from '../../../../app/hooks';
import { MCPTool, ToolStatus, ToolStatusInfo } from '../types';
import ToolStatusMonitor, { StatusHistory, AlertThresholds } from '../services/ToolStatusMonitor';
import { selectToolById, updateTool } from '../stores/toolsSlice';

interface UseToolStatusOptions {
  toolId?: string;
  toolIds?: string[];
  autoStart?: boolean;
  refreshInterval?: number;
  onStatusChange?: (toolId: string, oldStatus: ToolStatus, newStatus: ToolStatus) => void;
  onAlert?: (toolId: string, status: ToolStatus, message: string) => void;
}

interface UseToolStatusResult {
  // Status data
  status: ToolStatusInfo | null;
  statuses: Map<string, ToolStatusInfo>;
  history: StatusHistory[];
  histories: Map<string, StatusHistory[]>;
  
  // Monitoring state
  isMonitoring: boolean;
  lastUpdate: Date | null;
  
  // Actions
  startMonitoring: () => void;
  stopMonitoring: () => void;
  refreshStatus: () => Promise<void>;
  refreshAll: () => Promise<void>;
  
  // Configuration
  setThresholds: (thresholds: Partial<AlertThresholds>) => void;
  setRefreshInterval: (interval: number) => void;
  
  // Statistics
  getStats: () => {
    averageResponseTime: number;
    errorRate: number;
    availability: number;
    healthDistribution: Record<ToolStatus, number>;
  };
}

export function useToolStatus(options: UseToolStatusOptions = {}): UseToolStatusResult {
  const {
    toolId,
    toolIds = [],
    autoStart = true,
    refreshInterval = 30000,
    onStatusChange,
    onAlert
  } = options;

  const dispatch = useAppDispatch();
  const tool = useAppSelector(state => toolId ? selectToolById(toolId)(state) : null);
  
  // State
  const [status, setStatus] = useState<ToolStatusInfo | null>(tool?.status || null);
  const [statuses, setStatuses] = useState<Map<string, ToolStatusInfo>>(new Map());
  const [history, setHistory] = useState<StatusHistory[]>([]);
  const [histories, setHistories] = useState<Map<string, StatusHistory[]>>(new Map());
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  
  // Refs for callbacks
  const statusChangeRef = useRef(onStatusChange);
  const alertRef = useRef(onAlert);
  const intervalRef = useRef(refreshInterval);

  // Update refs when callbacks change
  useEffect(() => {
    statusChangeRef.current = onStatusChange;
    alertRef.current = onAlert;
    intervalRef.current = refreshInterval;
  }, [onStatusChange, onAlert, refreshInterval]);

  // Get all tool IDs to monitor
  const allToolIds = toolId ? [toolId, ...toolIds] : toolIds;

  // Load initial data
  useEffect(() => {
    if (allToolIds.length === 0) return;

    const loadInitialData = () => {
      // Load current statuses
      const newStatuses = new Map<string, ToolStatusInfo>();
      allToolIds.forEach(id => {
        const toolData = tool; // In real app, would fetch from store
        if (toolData?.status) {
          newStatuses.set(id, toolData.status);
        }
      });
      setStatuses(newStatuses);

      // Load histories
      const newHistories = new Map<string, StatusHistory[]>();
      allToolIds.forEach(id => {
        const toolHistory = ToolStatusMonitor.getStatusHistory(id, 24);
        newHistories.set(id, toolHistory);
      });
      setHistories(newHistories);

      // Set single tool data if applicable
      if (toolId && tool) {
        setStatus(tool.status);
        setHistory(ToolStatusMonitor.getStatusHistory(toolId, 24));
      }

      setLastUpdate(new Date());
    };

    loadInitialData();
  }, [allToolIds.join(','), toolId, tool]);

  // Set up monitoring
  useEffect(() => {
    if (!autoStart || allToolIds.length === 0) return;

    let cleanupFunctions: (() => void)[] = [];

    const startMonitoringTools = () => {
      // Start monitoring
      ToolStatusMonitor.startMonitoring(allToolIds, intervalRef.current);
      setIsMonitoring(true);

      // Set up status change listener
      const unsubscribeStatusChange = ToolStatusMonitor.onStatusChange((id, oldStatus, newStatus) => {
        if (allToolIds.includes(id)) {
          // Update local state
          setStatuses(prev => {
            const updated = new Map(prev);
            const toolData = { ...prev.get(id)!, health: newStatus };
            updated.set(id, toolData);
            return updated;
          });

          if (id === toolId) {
            setStatus(prev => prev ? { ...prev, health: newStatus } : null);
          }

          // Call user callback
          statusChangeRef.current?.(id, oldStatus, newStatus);
        }
      });

      // Set up alert listener
      const unsubscribeAlert = ToolStatusMonitor.onAlert((id, status, message) => {
        if (allToolIds.includes(id)) {
          alertRef.current?.(id, status, message);
        }
      });

      cleanupFunctions = [unsubscribeStatusChange, unsubscribeAlert];

      // Set up periodic history updates
      const historyInterval = setInterval(() => {
        const newHistories = new Map<string, StatusHistory[]>();
        allToolIds.forEach(id => {
          const toolHistory = ToolStatusMonitor.getStatusHistory(id, 24);
          newHistories.set(id, toolHistory);
        });
        setHistories(newHistories);

        if (toolId) {
          setHistory(ToolStatusMonitor.getStatusHistory(toolId, 24));
        }

        setLastUpdate(new Date());
      }, 60000); // Update history every minute

      cleanupFunctions.push(() => clearInterval(historyInterval));
    };

    startMonitoringTools();

    return () => {
      ToolStatusMonitor.stopMonitoring(allToolIds);
      setIsMonitoring(false);
      cleanupFunctions.forEach(cleanup => cleanup());
    };
  }, [allToolIds.join(','), autoStart]);

  // Actions
  const startMonitoring = useCallback(() => {
    if (!isMonitoring && allToolIds.length > 0) {
      ToolStatusMonitor.startMonitoring(allToolIds, intervalRef.current);
      setIsMonitoring(true);
    }
  }, [allToolIds, isMonitoring]);

  const stopMonitoring = useCallback(() => {
    if (isMonitoring && allToolIds.length > 0) {
      ToolStatusMonitor.stopMonitoring(allToolIds);
      setIsMonitoring(false);
    }
  }, [allToolIds, isMonitoring]);

  const refreshStatus = useCallback(async () => {
    if (!toolId || !tool) return;

    try {
      const newStatus = await ToolStatusMonitor.checkToolHealth(tool);
      setStatus(newStatus);
      dispatch(updateTool({ id: toolId, updates: { status: newStatus } }));
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to refresh tool status:', error);
    }
  }, [toolId, tool, dispatch]);

  const refreshAll = useCallback(async () => {
    for (const id of allToolIds) {
      // In real app, would fetch tool from store
      if (tool && id === toolId) {
        try {
          const newStatus = await ToolStatusMonitor.checkToolHealth(tool);
          setStatuses(prev => {
            const updated = new Map(prev);
            updated.set(id, newStatus);
            return updated;
          });
          
          if (id === toolId) {
            setStatus(newStatus);
          }
          
          dispatch(updateTool({ id, updates: { status: newStatus } }));
        } catch (error) {
          console.error(`Failed to refresh status for tool ${id}:`, error);
        }
      }
    }
    setLastUpdate(new Date());
  }, [allToolIds, tool, toolId, dispatch]);

  const setThresholds = useCallback((thresholds: Partial<AlertThresholds>) => {
    ToolStatusMonitor.setAlertThresholds(thresholds);
  }, []);

  const setRefreshInterval = useCallback((interval: number) => {
    intervalRef.current = interval;
    if (isMonitoring) {
      // Restart monitoring with new interval
      ToolStatusMonitor.stopMonitoring(allToolIds);
      ToolStatusMonitor.startMonitoring(allToolIds, interval);
    }
  }, [allToolIds, isMonitoring]);

  const getStats = useCallback(() => {
    const stats = {
      averageResponseTime: 0,
      errorRate: 0,
      availability: 0,
      healthDistribution: {
        healthy: 0,
        degraded: 0,
        unavailable: 0,
        unknown: 0
      } as Record<ToolStatus, number>
    };

    let totalResponseTime = 0;
    let totalErrorRate = 0;
    let availableCount = 0;
    let count = 0;

    statuses.forEach((status) => {
      count++;
      totalResponseTime += status.responseTime;
      totalErrorRate += status.errorRate;
      if (status.available) availableCount++;
      stats.healthDistribution[status.health]++;
    });

    if (count > 0) {
      stats.averageResponseTime = totalResponseTime / count;
      stats.errorRate = totalErrorRate / count;
      stats.availability = (availableCount / count) * 100;
    }

    return stats;
  }, [statuses]);

  return {
    // Status data
    status,
    statuses,
    history,
    histories,
    
    // Monitoring state
    isMonitoring,
    lastUpdate,
    
    // Actions
    startMonitoring,
    stopMonitoring,
    refreshStatus,
    refreshAll,
    
    // Configuration
    setThresholds,
    setRefreshInterval,
    
    // Statistics
    getStats
  };
}

// Hook for monitoring all tools
export function useAllToolsStatus(options: Omit<UseToolStatusOptions, 'toolId' | 'toolIds'> = {}) {
  const allTools = useAppSelector(state => state.tools.tools);
  const toolIds = allTools.map(t => t.id);
  
  return useToolStatus({
    ...options,
    toolIds
  });
}

// Hook for monitoring tools by category
export function useToolStatusByCategory(
  category: string,
  options: Omit<UseToolStatusOptions, 'toolId' | 'toolIds'> = {}
) {
  const tools = useAppSelector(state => 
    state.tools.tools.filter(t => t.category === category)
  );
  const toolIds = tools.map(t => t.id);
  
  return useToolStatus({
    ...options,
    toolIds
  });
}

// Hook for monitoring degraded/unavailable tools
export function useProblematicToolsStatus(options: Omit<UseToolStatusOptions, 'toolId' | 'toolIds'> = {}) {
  const problematicTools = useAppSelector(state => 
    state.tools.tools.filter(t => 
      t.status.health === 'degraded' || t.status.health === 'unavailable'
    )
  );
  const toolIds = problematicTools.map(t => t.id);
  
  return useToolStatus({
    ...options,
    toolIds
  });
}