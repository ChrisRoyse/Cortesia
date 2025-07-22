import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Button,
  Stack,
  Chip,
  Alert,
  LinearProgress,
  Tooltip,
  Tabs,
  Tab,
  Divider,
  useTheme,
} from '@mui/material';
import {
  ContentCopy,
  Download,
  Fullscreen,
  FullscreenExit,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  Info,
  Code,
  DataObject,
  Timeline,
} from '@mui/icons-material';
import { ToolExecution, ExecutionStatus } from '../../types';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ExecutionResultProps {
  execution: ToolExecution;
  isExecuting: boolean;
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
      sx={{ py: 2 }}
    >
      {value === index && children}
    </Box>
  );
};

const ExecutionResult: React.FC<ExecutionResultProps> = ({
  execution,
  isExecuting,
}) => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const syntaxTheme = theme.palette.mode === 'dark' ? vscDarkPlus : vs;

  const statusIcon = useMemo(() => {
    switch (execution.status) {
      case 'success':
        return <CheckCircle color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'running':
        return <CircularProgress size={20} />;
      case 'cancelled':
        return <Warning color="warning" />;
      default:
        return <Info color="info" />;
    }
  }, [execution.status]);

  const executionTime = useMemo(() => {
    if (!execution.endTime) {
      return isExecuting ? `Running (${Math.round((Date.now() - execution.startTime) / 1000)}s)` : 'N/A';
    }
    return `${execution.endTime - execution.startTime}ms`;
  }, [execution, isExecuting]);

  const handleCopy = useCallback((content: any) => {
    const text = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
    navigator.clipboard.writeText(text);
  }, []);

  const handleDownload = useCallback((content: any, filename: string) => {
    const text = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const renderContent = useCallback((content: any) => {
    if (!content) return <Typography color="text.secondary">No content</Typography>;

    // Handle different content types
    if (typeof content === 'string') {
      // Check if it's JSON
      try {
        const parsed = JSON.parse(content);
        return (
          <SyntaxHighlighter
            language="json"
            style={syntaxTheme}
            customStyle={{
              margin: 0,
              fontSize: '0.875rem',
              maxHeight: isFullscreen ? 'calc(100vh - 200px)' : '400px',
              overflow: 'auto',
            }}
          >
            {JSON.stringify(parsed, null, 2)}
          </SyntaxHighlighter>
        );
      } catch {
        // Not JSON, render as plain text
        return (
          <SyntaxHighlighter
            language="text"
            style={syntaxTheme}
            customStyle={{
              margin: 0,
              fontSize: '0.875rem',
              maxHeight: isFullscreen ? 'calc(100vh - 200px)' : '400px',
              overflow: 'auto',
            }}
          >
            {content}
          </SyntaxHighlighter>
        );
      }
    }

    // Render objects/arrays as JSON
    return (
      <SyntaxHighlighter
        language="json"
        style={syntaxTheme}
        customStyle={{
          margin: 0,
          fontSize: '0.875rem',
          maxHeight: isFullscreen ? 'calc(100vh - 200px)' : '400px',
          overflow: 'auto',
        }}
      >
        {JSON.stringify(content, null, 2)}
      </SyntaxHighlighter>
    );
  }, [syntaxTheme, isFullscreen]);

  const renderMetadata = useCallback(() => {
    const metadata = execution.metadata || {};
    return (
      <Stack spacing={2}>
        <Typography variant="subtitle2">Execution Metadata</Typography>
        <Stack spacing={1}>
          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2" color="text.secondary">Tool</Typography>
            <Typography variant="body2">{execution.toolId}</Typography>
          </Stack>
          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2" color="text.secondary">Execution ID</Typography>
            <Typography variant="body2" fontFamily="monospace">{execution.id}</Typography>
          </Stack>
          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2" color="text.secondary">Start Time</Typography>
            <Typography variant="body2">
              {new Date(execution.startTime).toLocaleString()}
            </Typography>
          </Stack>
          {execution.endTime && (
            <Stack direction="row" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">End Time</Typography>
              <Typography variant="body2">
                {new Date(execution.endTime).toLocaleString()}
              </Typography>
            </Stack>
          )}
          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2" color="text.secondary">Duration</Typography>
            <Typography variant="body2">{executionTime}</Typography>
          </Stack>
          {Object.entries(metadata).map(([key, value]) => (
            <Stack key={key} direction="row" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">{key}</Typography>
              <Typography variant="body2">{String(value)}</Typography>
            </Stack>
          ))}
        </Stack>
      </Stack>
    );
  }, [execution, executionTime]);

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 2,
        height: isFullscreen ? '100vh' : 'auto',
        position: isFullscreen ? 'fixed' : 'relative',
        top: isFullscreen ? 0 : 'auto',
        left: isFullscreen ? 0 : 'auto',
        right: isFullscreen ? 0 : 'auto',
        bottom: isFullscreen ? 0 : 'auto',
        zIndex: isFullscreen ? theme.zIndex.modal : 'auto',
        bgcolor: 'background.paper',
      }}
    >
      <Stack spacing={2} height="100%">
        {/* Header */}
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Stack direction="row" spacing={2} alignItems="center">
            {statusIcon}
            <Typography variant="h6">Execution Result</Typography>
            <Chip
              label={execution.status}
              size="small"
              color={
                execution.status === 'success' ? 'success' :
                execution.status === 'error' ? 'error' :
                execution.status === 'cancelled' ? 'warning' :
                'default'
              }
            />
          </Stack>
          <Stack direction="row" spacing={1}>
            <Tooltip title="Copy">
              <IconButton
                size="small"
                onClick={() => handleCopy(execution.output || execution.error)}
              >
                <ContentCopy />
              </IconButton>
            </Tooltip>
            <Tooltip title="Download">
              <IconButton
                size="small"
                onClick={() => handleDownload(
                  execution,
                  `execution-${execution.id}.json`
                )}
              >
                <Download />
              </IconButton>
            </Tooltip>
            <Tooltip title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}>
              <IconButton
                size="small"
                onClick={() => setIsFullscreen(!isFullscreen)}
              >
                {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
              </IconButton>
            </Tooltip>
          </Stack>
        </Stack>

        {/* Progress */}
        {isExecuting && (
          <LinearProgress variant="indeterminate" />
        )}

        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
            <Tab
              label="Output"
              icon={<DataObject />}
              iconPosition="start"
            />
            <Tab
              label="Input"
              icon={<Code />}
              iconPosition="start"
            />
            <Tab
              label="Metadata"
              icon={<Timeline />}
              iconPosition="start"
            />
          </Tabs>
        </Box>

        {/* Content */}
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          <TabPanel value={activeTab} index={0}>
            {execution.error ? (
              <Alert severity="error" sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Error
                </Typography>
                <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                  {execution.error}
                </Typography>
              </Alert>
            ) : (
              renderContent(execution.output)
            )}
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            {renderContent(execution.input)}
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            {renderMetadata()}
          </TabPanel>
        </Box>
      </Stack>
    </Paper>
  );
};

// Add missing import
import { CircularProgress } from '@mui/material';

export default ExecutionResult;