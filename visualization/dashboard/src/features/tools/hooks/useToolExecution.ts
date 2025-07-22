import { useState, useCallback, useEffect, useRef } from 'react';
import { useAppDispatch, useAppSelector } from '../../../app/hooks';
import { ToolExecution, ExecutionStatus } from '../types';
import { ToolExecutor, ExecutionUpdate } from '../services/ToolExecutor';
import { selectToolById, updateToolStats } from '../stores/toolsSlice';
import { Subscription } from 'rxjs';

interface UseToolExecutionReturn {
  execute: (input: any) => Promise<ToolExecution>;
  cancel: (executionId: string) => void;
  isExecuting: boolean;
  currentExecution: ToolExecution | null;
  executionHistory: ToolExecution[];
  executionUpdates: ExecutionUpdate[];
  clearHistory: () => void;
  exportHistory: () => string;
  deleteFromHistory: (executionId: string) => void;
}

const MAX_HISTORY_SIZE = 50;

export function useToolExecution(toolId: string): UseToolExecutionReturn {
  const dispatch = useAppDispatch();
  const tool = useAppSelector(state => selectToolById(state, toolId));
  
  const [isExecuting, setIsExecuting] = useState(false);
  const [currentExecution, setCurrentExecution] = useState<ToolExecution | null>(null);
  const [executionHistory, setExecutionHistory] = useState<ToolExecution[]>([]);
  const [executionUpdates, setExecutionUpdates] = useState<ExecutionUpdate[]>([]);
  
  const executorRef = useRef<ToolExecutor | null>(null);
  const subscriptionRef = useRef<Subscription | null>(null);

  // Initialize executor
  useEffect(() => {
    if (!executorRef.current) {
      executorRef.current = new ToolExecutor();
    }

    // Load history from localStorage
    const savedHistory = localStorage.getItem(`tool-history-${toolId}`);
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        if (Array.isArray(parsed)) {
          setExecutionHistory(parsed.slice(0, MAX_HISTORY_SIZE));
        }
      } catch (error) {
        console.error('Failed to load execution history:', error);
      }
    }

    return () => {
      // Clean up subscriptions
      if (subscriptionRef.current) {
        subscriptionRef.current.unsubscribe();
      }
    };
  }, [toolId]);

  // Save history to localStorage when it changes
  useEffect(() => {
    if (executionHistory.length > 0) {
      localStorage.setItem(`tool-history-${toolId}`, JSON.stringify(executionHistory));
    }
  }, [executionHistory, toolId]);

  const execute = useCallback(async (input: any): Promise<ToolExecution> => {
    if (!tool || !executorRef.current) {
      throw new Error('Tool or executor not available');
    }

    setIsExecuting(true);
    setExecutionUpdates([]);

    try {
      // Execute tool
      const executionPromise = executorRef.current.executeTool(tool, input);
      
      // Get execution ID from the promise (this is a bit hacky but works for our mock)
      const execution = await executionPromise;
      setCurrentExecution(execution);

      // Subscribe to updates
      if (subscriptionRef.current) {
        subscriptionRef.current.unsubscribe();
      }

      try {
        const stream = executorRef.current.getExecutionStream(execution.id);
        subscriptionRef.current = stream.subscribe({
          next: (update) => {
            setExecutionUpdates(prev => [...prev, update]);
            
            // Update current execution status
            if (update.type === 'status') {
              setCurrentExecution(prev => prev ? {
                ...prev,
                status: update.data.status,
                error: update.data.error,
              } : null);
            }
          },
          complete: () => {
            setIsExecuting(false);
          },
        });
      } catch (error) {
        console.error('Failed to subscribe to execution stream:', error);
      }

      // Wait for execution to complete
      const result = await executionPromise;

      // Update tool stats
      if (result.status === 'success' && result.endTime && result.startTime) {
        dispatch(updateToolStats({
          toolId,
          responseTime: result.endTime - result.startTime,
        }));
      }

      // Add to history
      setExecutionHistory(prev => {
        const newHistory = [result, ...prev].slice(0, MAX_HISTORY_SIZE);
        return newHistory;
      });

      setIsExecuting(false);
      return result;

    } catch (error) {
      setIsExecuting(false);
      throw error;
    }
  }, [tool, toolId, dispatch]);

  const cancel = useCallback((executionId: string) => {
    if (executorRef.current) {
      executorRef.current.cancelExecution(executionId);
    }
  }, []);

  const clearHistory = useCallback(() => {
    setExecutionHistory([]);
    localStorage.removeItem(`tool-history-${toolId}`);
  }, [toolId]);

  const exportHistory = useCallback((): string => {
    const exportData = {
      toolId,
      toolName: tool?.name,
      exportedAt: new Date().toISOString(),
      history: executionHistory,
    };
    return JSON.stringify(exportData, null, 2);
  }, [toolId, tool, executionHistory]);

  const deleteFromHistory = useCallback((executionId: string) => {
    setExecutionHistory(prev => prev.filter(exec => exec.id !== executionId));
  }, []);

  return {
    execute,
    cancel,
    isExecuting,
    currentExecution,
    executionHistory,
    executionUpdates,
    clearHistory,
    exportHistory,
    deleteFromHistory,
  };
}