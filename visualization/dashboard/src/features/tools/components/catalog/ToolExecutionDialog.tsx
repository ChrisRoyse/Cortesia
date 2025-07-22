import React, { useState, useCallback, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  TextField,
  CircularProgress,
  Alert,
  AlertTitle,
  Paper,
  IconButton,
  Tooltip,
  Stack,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  useTheme,
} from '@mui/material';
import {
  Close as CloseIcon,
  PlayArrow as ExecuteIcon,
  ContentCopy as CopyIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  AccessTime as TimeIcon,
} from '@mui/icons-material';
import { MCPTool } from '../../types';
import { useTool } from '../../hooks/useToolRegistry';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ToolExecutionDialogProps {
  open: boolean;
  tool: MCPTool;
  onClose: () => void;
}

interface ExecutionResult {
  success: boolean;
  data?: any;
  error?: any;
  duration: number;
  timestamp: Date;
}

export const ToolExecutionDialog: React.FC<ToolExecutionDialogProps> = ({
  open,
  tool,
  onClose,
}) => {
  const theme = useTheme();
  const { execute, executing, history, lastExecution } = useTool(tool.id);
  
  const [inputJson, setInputJson] = useState('{}');
  const [inputError, setInputError] = useState<string | null>(null);
  const [result, setResult] = useState<ExecutionResult | null>(null);
  const [copied, setCopied] = useState(false);

  // Initialize input with schema defaults
  useEffect(() => {
    if (tool.schema.inputSchema.properties) {
      const defaultInput: any = {};
      Object.entries(tool.schema.inputSchema.properties).forEach(([key, prop]: [string, any]) => {
        if (prop.default !== undefined) {
          defaultInput[key] = prop.default;
        } else if (tool.schema.inputSchema.required?.includes(key)) {
          // Set default values for required fields based on type
          switch (prop.type) {
            case 'string':
              defaultInput[key] = '';
              break;
            case 'number':
              defaultInput[key] = 0;
              break;
            case 'boolean':
              defaultInput[key] = false;
              break;
            case 'array':
              defaultInput[key] = [];
              break;
            case 'object':
              defaultInput[key] = {};
              break;
          }
        }
      });
      setInputJson(JSON.stringify(defaultInput, null, 2));
    }
  }, [tool]);

  // Validate JSON input
  const validateInput = useCallback((value: string) => {
    try {
      JSON.parse(value);
      setInputError(null);
      return true;
    } catch (e) {
      setInputError(e instanceof Error ? e.message : 'Invalid JSON');
      return false;
    }
  }, []);

  // Handle input change
  const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setInputJson(value);
    validateInput(value);
  }, [validateInput]);

  // Execute tool
  const handleExecute = useCallback(async () => {
    if (!validateInput(inputJson)) return;

    const startTime = Date.now();
    try {
      const params = JSON.parse(inputJson);
      const response = await execute(params);
      const duration = Date.now() - startTime;
      
      setResult({
        success: true,
        data: response,
        duration,
        timestamp: new Date(),
      });
    } catch (error) {
      const duration = Date.now() - startTime;
      setResult({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        duration,
        timestamp: new Date(),
      });
    }
  }, [inputJson, validateInput, execute]);

  // Copy result to clipboard
  const handleCopyResult = useCallback(() => {
    if (result) {
      const toCopy = result.success ? result.data : result.error;
      navigator.clipboard.writeText(JSON.stringify(toCopy, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [result]);

  // Format execution time
  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="h6">Execute: {tool.name}</Typography>
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
        <Stack spacing={3}>
          {/* Tool description */}
          <Alert severity="info">
            <AlertTitle>Tool Description</AlertTitle>
            {tool.description}
          </Alert>

          {/* Input parameters */}
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              Input Parameters
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={10}
              value={inputJson}
              onChange={handleInputChange}
              error={!!inputError}
              helperText={inputError}
              variant="outlined"
              sx={{
                '& .MuiInputBase-root': {
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                },
              }}
            />
          </Box>

          {/* Input schema reference */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2">Input Schema Reference</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Paper variant="outlined" sx={{ p: 1, backgroundColor: theme.palette.grey[900] }}>
                <SyntaxHighlighter
                  language="json"
                  style={vscDarkPlus}
                  customStyle={{ margin: 0, fontSize: '0.75rem' }}
                >
                  {JSON.stringify(tool.schema.inputSchema, null, 2)}
                </SyntaxHighlighter>
              </Paper>
            </AccordionDetails>
          </Accordion>

          {/* Execution result */}
          {result && (
            <Box>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="subtitle1">
                  Execution Result
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  {result.success ? (
                    <Chip
                      icon={<SuccessIcon />}
                      label="Success"
                      color="success"
                      size="small"
                    />
                  ) : (
                    <Chip
                      icon={<ErrorIcon />}
                      label="Error"
                      color="error"
                      size="small"
                    />
                  )}
                  <Chip
                    icon={<TimeIcon />}
                    label={formatDuration(result.duration)}
                    size="small"
                    variant="outlined"
                  />
                  <Tooltip title={copied ? 'Copied!' : 'Copy result'}>
                    <IconButton size="small" onClick={handleCopyResult}>
                      <CopyIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              <Paper
                variant="outlined"
                sx={{
                  p: 2,
                  backgroundColor: result.success ? 'action.hover' : 'error.dark',
                  maxHeight: 400,
                  overflow: 'auto',
                }}
              >
                <SyntaxHighlighter
                  language="json"
                  style={vscDarkPlus}
                  customStyle={{ margin: 0, fontSize: '0.875rem' }}
                >
                  {JSON.stringify(result.success ? result.data : result.error, null, 2)}
                </SyntaxHighlighter>
              </Paper>
            </Box>
          )}

          {/* Execution history */}
          {history.length > 0 && (
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle2">
                  Recent Executions ({history.length})
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Stack spacing={1}>
                  {history.slice(-5).reverse().map((exec, index) => (
                    <Paper key={exec.id} variant="outlined" sx={{ p: 1.5 }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Box display="flex" alignItems="center" gap={1}>
                          {exec.status === 'success' ? (
                            <SuccessIcon color="success" fontSize="small" />
                          ) : exec.status === 'error' ? (
                            <ErrorIcon color="error" fontSize="small" />
                          ) : (
                            <CircularProgress size={16} />
                          )}
                          <Typography variant="caption">
                            {new Date(exec.startTime).toLocaleString()}
                          </Typography>
                        </Box>
                        {exec.duration && (
                          <Typography variant="caption" color="text.secondary">
                            {formatDuration(exec.duration)}
                          </Typography>
                        )}
                      </Box>
                      {exec.error && (
                        <Typography variant="caption" color="error" sx={{ mt: 0.5 }}>
                          Error: {exec.error.message}
                        </Typography>
                      )}
                    </Paper>
                  ))}
                </Stack>
              </AccordionDetails>
            </Accordion>
          )}
        </Stack>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        <Button
          variant="contained"
          startIcon={executing ? <CircularProgress size={20} /> : <ExecuteIcon />}
          onClick={handleExecute}
          disabled={executing || !!inputError}
        >
          {executing ? 'Executing...' : 'Execute'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ToolExecutionDialog;