import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  IconButton,
  Tooltip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  ViewColumn,
  Tab,
  CompareArrows,
  Fullscreen,
  FullscreenExit,
  ContentCopy,
  Download,
} from '@mui/icons-material';
import { MCPTool } from '../../types';
import JsonViewer from './JsonViewer';
import DiffViewer from './DiffViewer';
import GraphVisualization from './GraphVisualization';
import NeuralDataViewer from './NeuralDataViewer';
import { formatExecutionTime, copyToClipboard, exportAsJson } from '../../utils/dataFormatters';

interface RequestResponseViewProps {
  request: any;
  response: any;
  tool: MCPTool;
  executionTime: number;
  viewMode?: 'split' | 'tabs' | 'diff';
  onViewModeChange?: (mode: 'split' | 'tabs' | 'diff') => void;
}

const RequestResponseView: React.FC<RequestResponseViewProps> = ({
  request,
  response,
  tool,
  executionTime,
  viewMode = 'split',
  onViewModeChange,
}) => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState<'request' | 'response'>('request');
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleViewModeChange = (
    event: React.MouseEvent<HTMLElement>,
    newMode: 'split' | 'tabs' | 'diff' | null
  ) => {
    if (newMode && onViewModeChange) {
      onViewModeChange(newMode);
    }
  };

  const handleCopy = () => {
    const data = viewMode === 'diff' 
      ? { request, response }
      : activeTab === 'request' ? request : response;
    copyToClipboard(data);
  };

  const handleExport = () => {
    exportAsJson({
      tool: tool.name,
      request,
      response,
      executionTime,
      timestamp: new Date().toISOString(),
    }, `${tool.name}_execution_${Date.now()}.json`);
  };

  // Determine if response contains special LLMKG data types
  const hasGraphData = useMemo(() => {
    return response?.data?.nodes && response?.data?.edges;
  }, [response]);

  const hasNeuralData = useMemo(() => {
    return response?.data?.neural_activity || response?.data?.sdr_data;
  }, [response]);

  const renderContent = () => {
    if (hasGraphData && viewMode !== 'diff') {
      return (
        <GraphVisualization
          data={response.data}
          fullscreen={isFullscreen}
        />
      );
    }

    if (hasNeuralData && viewMode !== 'diff') {
      return (
        <NeuralDataViewer
          data={response.data}
          fullscreen={isFullscreen}
        />
      );
    }

    switch (viewMode) {
      case 'split':
        return (
          <Box sx={{ display: 'flex', gap: 2, height: '100%' }}>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                Request
              </Typography>
              <JsonViewer
                data={request}
                theme={theme.palette.mode}
                expandLevel={2}
              />
            </Box>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                Response
              </Typography>
              <JsonViewer
                data={response}
                theme={theme.palette.mode}
                expandLevel={2}
              />
            </Box>
          </Box>
        );

      case 'tabs':
        return (
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
              <ToggleButtonGroup
                value={activeTab}
                exclusive
                onChange={(e, value) => value && setActiveTab(value)}
                size="small"
              >
                <ToggleButton value="request">Request</ToggleButton>
                <ToggleButton value="response">Response</ToggleButton>
              </ToggleButtonGroup>
            </Box>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <JsonViewer
                data={activeTab === 'request' ? request : response}
                theme={theme.palette.mode}
                expandLevel={3}
              />
            </Box>
          </Box>
        );

      case 'diff':
        return (
          <DiffViewer
            before={request}
            after={response}
            beforeLabel="Request"
            afterLabel="Response"
          />
        );

      default:
        return null;
    }
  };

  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        height: isFullscreen ? '100vh' : 'auto',
        display: 'flex',
        flexDirection: 'column',
        ...(isFullscreen && {
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: theme.zIndex.modal,
          borderRadius: 0,
        }),
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 2,
          pb: 2,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h6">{tool.name}</Typography>
          <Chip
            label={formatExecutionTime(executionTime)}
            size="small"
            color="primary"
            variant="outlined"
          />
          {hasGraphData && (
            <Chip
              label="Graph Data"
              size="small"
              color="secondary"
              variant="filled"
            />
          )}
          {hasNeuralData && (
            <Chip
              label="Neural Data"
              size="small"
              color="secondary"
              variant="filled"
            />
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {!hasGraphData && !hasNeuralData && (
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={handleViewModeChange}
              size="small"
            >
              <ToggleButton value="split">
                <Tooltip title="Split View">
                  <ViewColumn />
                </Tooltip>
              </ToggleButton>
              <ToggleButton value="tabs">
                <Tooltip title="Tab View">
                  <Tab />
                </Tooltip>
              </ToggleButton>
              <ToggleButton value="diff">
                <Tooltip title="Diff View">
                  <CompareArrows />
                </Tooltip>
              </ToggleButton>
            </ToggleButtonGroup>
          )}

          <Tooltip title="Copy">
            <IconButton size="small" onClick={handleCopy}>
              <ContentCopy />
            </IconButton>
          </Tooltip>

          <Tooltip title="Export">
            <IconButton size="small" onClick={handleExport}>
              <Download />
            </IconButton>
          </Tooltip>

          <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
            <IconButton
              size="small"
              onClick={() => setIsFullscreen(!isFullscreen)}
            >
              {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Content */}
      <Box sx={{ flex: 1, overflow: 'auto', minHeight: 400 }}>
        {renderContent()}
      </Box>
    </Paper>
  );
};

export default RequestResponseView;