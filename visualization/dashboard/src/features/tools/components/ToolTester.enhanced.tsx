import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { PlayArrow } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../../hooks/redux';
import { executeTool, selectToolExecution } from '../store/toolsSlice';
import { MCPTool } from '../types';
import ToolForm from './ToolForm';
import RequestResponseView from './visualization/RequestResponseView';

interface ToolTesterProps {
  tool: MCPTool;
}

const ToolTester: React.FC<ToolTesterProps> = ({ tool }) => {
  const dispatch = useAppDispatch();
  const { isExecuting, lastResult, lastError } = useAppSelector(selectToolExecution);
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [viewMode, setViewMode] = useState<'split' | 'tabs' | 'diff'>('split');
  const [executionHistory, setExecutionHistory] = useState<Array<{
    request: any;
    response: any;
    timestamp: number;
    executionTime: number;
  }>>([]);
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState<number>(-1);

  const handleParameterChange = (name: string, value: any) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleExecute = async () => {
    const startTime = Date.now();
    
    try {
      const result = await dispatch(
        executeTool({ toolName: tool.name, parameters })
      ).unwrap();
      
      const executionTime = Date.now() - startTime;
      
      // Add to history
      const historyEntry = {
        request: parameters,
        response: result,
        timestamp: Date.now(),
        executionTime,
      };
      
      setExecutionHistory((prev) => [...prev, historyEntry]);
      setSelectedHistoryIndex(executionHistory.length);
    } catch (error) {
      console.error('Tool execution failed:', error);
    }
  };

  const currentExecution = selectedHistoryIndex >= 0 
    ? executionHistory[selectedHistoryIndex]
    : executionHistory[executionHistory.length - 1];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          {tool.name}
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          {tool.description}
        </Typography>

        <Divider sx={{ my: 2 }} />

        <ToolForm
          tool={tool}
          parameters={parameters}
          onParameterChange={handleParameterChange}
        />

        <Box sx={{ mt: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={isExecuting ? <CircularProgress size={20} /> : <PlayArrow />}
            onClick={handleExecute}
            disabled={isExecuting}
          >
            {isExecuting ? 'Executing...' : 'Execute Tool'}
          </Button>

          {executionHistory.length > 0 && (
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Execution History</InputLabel>
              <Select
                value={selectedHistoryIndex}
                onChange={(e) => setSelectedHistoryIndex(Number(e.target.value))}
                label="Execution History"
              >
                {executionHistory.map((entry, index) => (
                  <MenuItem key={index} value={index}>
                    {new Date(entry.timestamp).toLocaleTimeString()} - {entry.executionTime}ms
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
        </Box>

        {lastError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {lastError}
          </Alert>
        )}
      </Paper>

      {currentExecution && (
        <RequestResponseView
          request={currentExecution.request}
          response={currentExecution.response}
          tool={tool}
          executionTime={currentExecution.executionTime}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
        />
      )}

      {/* Example of specialized visualizations based on tool type */}
      {tool.name === 'graph_query' && lastResult?.data?.nodes && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            Graph Visualization
          </Typography>
          <Typography variant="body2" color="text.secondary">
            The graph visualization is automatically displayed in the Request/Response view above
            when graph data is detected in the response.
          </Typography>
        </Paper>
      )}

      {tool.name === 'neural_activity' && lastResult?.data?.neural_activity && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            Neural Activity Visualization
          </Typography>
          <Typography variant="body2" color="text.secondary">
            The neural activity heatmap and spike trains are automatically displayed in the 
            Request/Response view above when neural data is detected in the response.
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default ToolTester;