import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  Button,
  CircularProgress,
  Alert,
  Divider,
  IconButton,
  Tooltip,
  Chip,
  Stack,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  History,
  Code,
  ContentCopy,
  Download,
  Refresh,
  Settings,
} from '@mui/icons-material';
import { MCPTool, ToolExecution } from '../../types';
import DynamicForm from './DynamicForm';
import ExecutionResult from './ExecutionResult';
import ExecutionHistory from './ExecutionHistory';
import { useToolExecution } from '../../hooks/useToolExecution';
import { useAppSelector } from '../../../../app/hooks';
import { selectToolById } from '../../stores/toolsSlice';

interface ToolTesterProps {
  toolId: string;
  onClose?: () => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <Box
      role="tabpanel"
      hidden={value !== index}
      id={`tool-tabpanel-${index}`}
      aria-labelledby={`tool-tab-${index}`}
      sx={{ py: 2 }}
    >
      {value === index && children}
    </Box>
  );
};

const ToolTester: React.FC<ToolTesterProps> = ({ toolId, onClose }) => {
  const tool = useAppSelector(state => selectToolById(state, toolId));
  const {
    execute,
    cancel,
    isExecuting,
    currentExecution,
    executionHistory,
    clearHistory,
    exportHistory,
  } = useToolExecution(toolId);

  const [activeTab, setActiveTab] = useState(0);
  const [formValues, setFormValues] = useState<any>({});
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<string | null>(null);

  const handleTabChange = useCallback((event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  }, []);

  const handleExecute = useCallback(async (values: any) => {
    setFormValues(values);
    await execute(values);
    // Switch to results tab after execution starts
    setActiveTab(1);
  }, [execute]);

  const handleCancel = useCallback(() => {
    if (currentExecution?.id) {
      cancel(currentExecution.id);
    }
  }, [cancel, currentExecution]);

  const handleCopyInput = useCallback(() => {
    navigator.clipboard.writeText(JSON.stringify(formValues, null, 2));
  }, [formValues]);

  const handleLoadFromHistory = useCallback((execution: ToolExecution) => {
    setFormValues(execution.input);
    setSelectedHistoryItem(execution.id);
    setActiveTab(0);
  }, []);

  const handleExportHistory = useCallback(() => {
    const data = exportHistory();
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${tool?.name || 'tool'}-history-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [exportHistory, tool]);

  const toolStatus = useMemo(() => {
    if (!tool) return null;
    
    return (
      <Stack direction="row" spacing={1} alignItems="center">
        <Chip
          label={tool.status}
          size="small"
          color={tool.status === 'available' ? 'success' : 'warning'}
        />
        {tool.responseTime && (
          <Typography variant="caption" color="text.secondary">
            {tool.responseTime}ms avg
          </Typography>
        )}
      </Stack>
    );
  }, [tool]);

  if (!tool) {
    return (
      <Alert severity="error">
        Tool not found
      </Alert>
    );
  }

  return (
    <Paper sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ mb: 2 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h5" gutterBottom>
              {tool.name}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {tool.description}
            </Typography>
            {toolStatus}
          </Box>
          <Stack direction="row" spacing={1}>
            <Tooltip title="Export History">
              <IconButton onClick={handleExportHistory} size="small">
                <Download />
              </IconButton>
            </Tooltip>
            <Tooltip title="Clear History">
              <IconButton onClick={clearHistory} size="small">
                <Refresh />
              </IconButton>
            </Tooltip>
            {onClose && (
              <Button onClick={onClose} size="small">
                Close
              </Button>
            )}
          </Stack>
        </Stack>
      </Box>

      <Divider sx={{ mb: 2 }} />

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleTabChange}>
          <Tab 
            label="Input" 
            icon={<Code />} 
            iconPosition="start"
          />
          <Tab 
            label="Result" 
            icon={<PlayArrow />} 
            iconPosition="start"
            disabled={!currentExecution && executionHistory.length === 0}
          />
          <Tab 
            label={`History (${executionHistory.length})`} 
            icon={<History />} 
            iconPosition="start"
          />
        </Tabs>
      </Box>

      {/* Content */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <TabPanel value={activeTab} index={0}>
          <Stack spacing={2}>
            {/* Action Buttons */}
            <Stack direction="row" spacing={2} alignItems="center">
              {isExecuting ? (
                <Button
                  variant="contained"
                  color="error"
                  startIcon={<Stop />}
                  onClick={handleCancel}
                >
                  Stop Execution
                </Button>
              ) : (
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<PlayArrow />}
                  onClick={() => handleExecute(formValues)}
                  disabled={tool.status !== 'available'}
                >
                  Execute
                </Button>
              )}
              <Button
                variant="outlined"
                startIcon={<ContentCopy />}
                onClick={handleCopyInput}
                size="small"
              >
                Copy Input
              </Button>
              {selectedHistoryItem && (
                <Typography variant="caption" color="text.secondary">
                  Loaded from history
                </Typography>
              )}
            </Stack>

            {/* Dynamic Form */}
            <DynamicForm
              schema={tool.inputSchema}
              onSubmit={handleExecute}
              onChange={setFormValues}
              initialValues={formValues}
              examples={tool.examples}
              disabled={isExecuting}
            />
          </Stack>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          {currentExecution ? (
            <ExecutionResult
              execution={currentExecution}
              isExecuting={isExecuting}
            />
          ) : executionHistory.length > 0 ? (
            <ExecutionResult
              execution={executionHistory[0]}
              isExecuting={false}
            />
          ) : (
            <Alert severity="info">
              No execution results yet. Execute the tool to see results.
            </Alert>
          )}
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <ExecutionHistory
            history={executionHistory}
            onSelect={handleLoadFromHistory}
            selectedId={selectedHistoryItem}
          />
        </TabPanel>
      </Box>

      {/* Status Bar */}
      {isExecuting && currentExecution && (
        <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
          <Stack direction="row" spacing={2} alignItems="center">
            <CircularProgress size={20} />
            <Typography variant="body2">
              Executing... ({Math.round((Date.now() - currentExecution.startTime) / 1000)}s)
            </Typography>
          </Stack>
        </Box>
      )}
    </Paper>
  );
};

export default ToolTester;