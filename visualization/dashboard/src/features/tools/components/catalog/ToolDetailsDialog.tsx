import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Chip,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Stack,
  Alert,
  LinearProgress,
  useTheme,
} from '@mui/material';
import {
  Close as CloseIcon,
  PlayArrow as ExecuteIcon,
  ContentCopy as CopyIcon,
  CheckCircle as HealthyIcon,
  Warning as DegradedIcon,
  Error as UnavailableIcon,
  Help as UnknownIcon,
  Speed as SpeedIcon,
  Functions as FunctionsIcon,
  Code as CodeIcon,
} from '@mui/icons-material';
import { MCPTool, ToolStatus } from '../../types';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ToolDetailsDialogProps {
  open: boolean;
  tool: MCPTool;
  onClose: () => void;
  onExecute: () => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
}

const statusConfig: Record<ToolStatus, { icon: React.ReactNode; color: string; label: string }> = {
  healthy: { icon: <HealthyIcon />, color: '#4CAF50', label: 'Healthy' },
  degraded: { icon: <DegradedIcon />, color: '#FF9800', label: 'Degraded' },
  unavailable: { icon: <UnavailableIcon />, color: '#F44336', label: 'Unavailable' },
  unknown: { icon: <UnknownIcon />, color: '#9E9E9E', label: 'Unknown' },
};

export const ToolDetailsDialog: React.FC<ToolDetailsDialogProps> = ({
  open,
  tool,
  onClose,
  onExecute,
}) => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [copiedSchema, setCopiedSchema] = useState(false);

  const handleCopySchema = (schema: any) => {
    navigator.clipboard.writeText(JSON.stringify(schema, null, 2));
    setCopiedSchema(true);
    setTimeout(() => setCopiedSchema(false), 2000);
  };

  const statusInfo = statusConfig[tool.status.health];

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="h6">{tool.name}</Typography>
            <Typography variant="caption" color="text.secondary">
              v{tool.version}
            </Typography>
            <Chip
              label={tool.category}
              size="small"
              color="primary"
            />
          </Box>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        <Stack spacing={2}>
          {/* Status and metrics summary */}
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Box display="flex" alignItems="center" gap={1}>
                <Box sx={{ color: statusInfo.color }}>
                  {statusInfo.icon}
                </Box>
                <Typography variant="body1" fontWeight="medium">
                  Status: {statusInfo.label}
                </Typography>
              </Box>
              <Box display="flex" gap={3}>
                <Tooltip title="Average response time">
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <SpeedIcon fontSize="small" />
                    <Typography variant="body2">
                      {tool.metrics.averageResponseTime.toFixed(0)}ms
                    </Typography>
                  </Box>
                </Tooltip>
                <Tooltip title="Success rate">
                  <Typography
                    variant="body2"
                    sx={{
                      color: tool.metrics.successRate > 95 ? '#4CAF50' : tool.metrics.successRate > 80 ? '#FF9800' : '#F44336',
                    }}
                  >
                    {tool.metrics.successRate.toFixed(1)}% success
                  </Typography>
                </Tooltip>
                <Tooltip title="Total executions">
                  <Typography variant="body2">
                    {tool.metrics.totalExecutions.toLocaleString()} executions
                  </Typography>
                </Tooltip>
              </Box>
            </Box>
            {tool.status.message && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                {tool.status.message}
              </Typography>
            )}
          </Paper>

          {/* Description */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Description
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {tool.description}
            </Typography>
          </Box>

          {/* Tags */}
          {tool.tags && tool.tags.length > 0 && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Tags
              </Typography>
              <Box display="flex" gap={0.5} flexWrap="wrap">
                {tool.tags.map(tag => (
                  <Chip key={tag} label={tag} size="small" variant="outlined" />
                ))}
              </Box>
            </Box>
          )}

          {/* Tabs */}
          <Box>
            <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)}>
              <Tab label="Schema" />
              <Tab label="Documentation" />
              <Tab label="Performance" />
              <Tab label="Examples" />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
              {/* Schema tab */}
              <Stack spacing={2}>
                <Box>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="subtitle2">Input Schema</Typography>
                    <Tooltip title={copiedSchema ? 'Copied!' : 'Copy schema'}>
                      <IconButton size="small" onClick={() => handleCopySchema(tool.schema.inputSchema)}>
                        <CopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Paper variant="outlined" sx={{ p: 1, backgroundColor: theme.palette.grey[900] }}>
                    <SyntaxHighlighter
                      language="json"
                      style={vscDarkPlus}
                      customStyle={{ margin: 0, fontSize: '0.875rem' }}
                    >
                      {JSON.stringify(tool.schema.inputSchema, null, 2)}
                    </SyntaxHighlighter>
                  </Paper>
                </Box>

                <Box>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="subtitle2">Output Schema</Typography>
                    <Tooltip title="Copy schema">
                      <IconButton size="small" onClick={() => handleCopySchema(tool.schema.outputSchema)}>
                        <CopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Paper variant="outlined" sx={{ p: 1, backgroundColor: theme.palette.grey[900] }}>
                    <SyntaxHighlighter
                      language="json"
                      style={vscDarkPlus}
                      customStyle={{ margin: 0, fontSize: '0.875rem' }}
                    >
                      {JSON.stringify(tool.schema.outputSchema, null, 2)}
                    </SyntaxHighlighter>
                  </Paper>
                </Box>
              </Stack>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              {/* Documentation tab */}
              <Stack spacing={2}>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Summary
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {tool.documentation.summary}
                  </Typography>
                </Box>

                {tool.documentation.parameters.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Parameters
                    </Typography>
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Name</TableCell>
                            <TableCell>Type</TableCell>
                            <TableCell>Required</TableCell>
                            <TableCell>Description</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {tool.documentation.parameters.map(param => (
                            <TableRow key={param.name}>
                              <TableCell>{param.name}</TableCell>
                              <TableCell>
                                <Chip label={param.type} size="small" variant="outlined" />
                              </TableCell>
                              <TableCell>
                                {param.required ? (
                                  <Chip label="Required" size="small" color="error" />
                                ) : (
                                  <Chip label="Optional" size="small" />
                                )}
                              </TableCell>
                              <TableCell>{param.description}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                )}

                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Returns
                  </Typography>
                  <Alert severity="info" icon={<CodeIcon />}>
                    <Typography variant="body2">
                      Type: <strong>{tool.documentation.returns.type}</strong>
                    </Typography>
                    <Typography variant="caption">
                      {tool.documentation.returns.description}
                    </Typography>
                  </Alert>
                </Box>

                {tool.documentation.relatedTools && tool.documentation.relatedTools.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Related Tools
                    </Typography>
                    <Box display="flex" gap={1} flexWrap="wrap">
                      {tool.documentation.relatedTools.map(related => (
                        <Chip key={related} label={related} size="small" />
                      ))}
                    </Box>
                  </Box>
                )}
              </Stack>
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              {/* Performance tab */}
              <Stack spacing={2}>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Response Time Distribution
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Stack spacing={1}>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">Average</Typography>
                        <Typography variant="body2" fontWeight="medium">
                          {tool.metrics.averageResponseTime.toFixed(0)}ms
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">P95</Typography>
                        <Typography variant="body2" fontWeight="medium">
                          {tool.metrics.p95ResponseTime.toFixed(0)}ms
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">P99</Typography>
                        <Typography variant="body2" fontWeight="medium">
                          {tool.metrics.p99ResponseTime.toFixed(0)}ms
                        </Typography>
                      </Box>
                    </Stack>
                  </Paper>
                </Box>

                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Reliability Metrics
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Stack spacing={2}>
                      <Box>
                        <Box display="flex" justifyContent="space-between" mb={1}>
                          <Typography variant="body2">Success Rate</Typography>
                          <Typography variant="body2" fontWeight="medium">
                            {tool.metrics.successRate.toFixed(2)}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={tool.metrics.successRate}
                          sx={{
                            backgroundColor: 'action.hover',
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: tool.metrics.successRate > 95 ? '#4CAF50' : tool.metrics.successRate > 80 ? '#FF9800' : '#F44336',
                            },
                          }}
                        />
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">Total Errors</Typography>
                        <Typography variant="body2" fontWeight="medium" color="error">
                          {tool.metrics.errorCount}
                        </Typography>
                      </Box>
                      {tool.metrics.lastExecutionTime && (
                        <Box display="flex" justifyContent="space-between">
                          <Typography variant="body2">Last Execution</Typography>
                          <Typography variant="body2" fontWeight="medium">
                            {new Date(tool.metrics.lastExecutionTime).toLocaleString()}
                          </Typography>
                        </Box>
                      )}
                    </Stack>
                  </Paper>
                </Box>

                {Object.keys(tool.metrics.errorTypes).length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Error Breakdown
                    </Typography>
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Error Type</TableCell>
                            <TableCell align="right">Count</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(tool.metrics.errorTypes).map(([type, count]) => (
                            <TableRow key={type}>
                              <TableCell>{type}</TableCell>
                              <TableCell align="right">{count}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                )}
              </Stack>
            </TabPanel>

            <TabPanel value={tabValue} index={3}>
              {/* Examples tab */}
              <Stack spacing={2}>
                {tool.documentation.examples.length > 0 ? (
                  tool.documentation.examples.map((example, index) => (
                    <Box key={index}>
                      <Typography variant="subtitle2" gutterBottom>
                        Example {index + 1}: {example.language}
                      </Typography>
                      {example.description && (
                        <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                          {example.description}
                        </Typography>
                      )}
                      <Paper variant="outlined" sx={{ p: 1, backgroundColor: theme.palette.grey[900] }}>
                        <SyntaxHighlighter
                          language={example.language}
                          style={vscDarkPlus}
                          customStyle={{ margin: 0, fontSize: '0.875rem' }}
                        >
                          {example.code}
                        </SyntaxHighlighter>
                      </Paper>
                    </Box>
                  ))
                ) : (
                  <Alert severity="info">
                    No examples available for this tool yet.
                  </Alert>
                )}
              </Stack>
            </TabPanel>
          </Box>
        </Stack>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        <Button
          variant="contained"
          startIcon={<ExecuteIcon />}
          onClick={onExecute}
        >
          Execute Tool
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ToolDetailsDialog;