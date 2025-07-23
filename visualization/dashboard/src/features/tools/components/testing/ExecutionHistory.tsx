import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Stack,
  Chip,
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  Divider,
  Alert,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
} from '@mui/material';
import {
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  Cancel,
  Search,
  FilterList,
  Delete,
  ContentCopy,
  MoreVert,
  AccessTime,
  Speed,
  DataObject,
} from '@mui/icons-material';
import { ToolExecution, ExecutionStatus } from '../../types';

interface ExecutionHistoryProps {
  history: ToolExecution[];
  onSelect: (execution: ToolExecution) => void;
  selectedId?: string | null;
  onDelete?: (executionId: string) => void;
}

interface FilterState {
  search: string;
  status: ExecutionStatus | 'all';
  sortBy: 'time' | 'duration' | 'status';
  sortOrder: 'asc' | 'desc';
}

const ExecutionHistory: React.FC<ExecutionHistoryProps> = ({
  history,
  onSelect,
  selectedId,
  onDelete,
}) => {
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    status: 'all',
    sortBy: 'time',
    sortOrder: 'desc',
  });

  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedExecution, setSelectedExecution] = useState<ToolExecution | null>(null);

  const handleFilterChange = useCallback((key: keyof FilterState, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleMenuClick = useCallback((event: React.MouseEvent<HTMLElement>, execution: ToolExecution) => {
    event.stopPropagation();
    setAnchorEl(event.currentTarget);
    setSelectedExecution(execution);
  }, []);

  const handleMenuClose = useCallback(() => {
    setAnchorEl(null);
    setSelectedExecution(null);
  }, []);

  const handleCopy = useCallback((execution: ToolExecution) => {
    navigator.clipboard.writeText(JSON.stringify(execution, null, 2));
    handleMenuClose();
  }, []);

  const handleDelete = useCallback((execution: ToolExecution) => {
    if (onDelete) {
      onDelete(execution.id);
    }
    handleMenuClose();
  }, [onDelete]);

  const filteredAndSortedHistory = useMemo(() => {
    let filtered = history;

    // Apply search filter
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(exec => 
        JSON.stringify(exec.input).toLowerCase().includes(searchLower) ||
        JSON.stringify(exec.output).toLowerCase().includes(searchLower) ||
        exec.error?.toLowerCase().includes(searchLower) ||
        exec.id.toLowerCase().includes(searchLower)
      );
    }

    // Apply status filter
    if (filters.status !== 'all') {
      filtered = filtered.filter(exec => exec.status === filters.status);
    }

    // Sort
    const sorted = [...filtered].sort((a, b) => {
      let comparison = 0;

      switch (filters.sortBy) {
        case 'time':
          comparison = a.startTime - b.startTime;
          break;
        case 'duration':
          const durationA = (a.endTime || Date.now()) - a.startTime;
          const durationB = (b.endTime || Date.now()) - b.startTime;
          comparison = durationA - durationB;
          break;
        case 'status':
          comparison = a.status.localeCompare(b.status);
          break;
      }

      return filters.sortOrder === 'asc' ? comparison : -comparison;
    });

    return sorted;
  }, [history, filters]);

  const getStatusIcon = (status: ExecutionStatus) => {
    switch (status) {
      case 'success':
        return <CheckCircle color="success" fontSize="small" />;
      case 'error':
        return <ErrorIcon color="error" fontSize="small" />;
      case 'cancelled':
        return <Cancel color="warning" fontSize="small" />;
      default:
        return <Warning color="info" fontSize="small" />;
    }
  };

  const formatDuration = (execution: ToolExecution) => {
    if (!execution.endTime) return 'Running...';
    const duration = execution.endTime - execution.startTime;
    if (duration < 1000) return `${duration}ms`;
    return `${(duration / 1000).toFixed(1)}s`;
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
  };

  if (history.length === 0) {
    return (
      <Alert severity="info">
        No execution history yet. Run the tool to see results here.
      </Alert>
    );
  }

  return (
    <Box>
      {/* Filters */}
      <Stack spacing={2} sx={{ mb: 2 }}>
        <TextField
          fullWidth
          size="small"
          placeholder="Search in history..."
          value={filters.search}
          onChange={(e) => handleFilterChange('search', e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
        />

        <Stack direction="row" spacing={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Status</InputLabel>
            <Select
              value={filters.status}
              label="Status"
              onChange={(e: SelectChangeEvent) => handleFilterChange('status', e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="success">Success</MenuItem>
              <MenuItem value="error">Error</MenuItem>
              <MenuItem value="cancelled">Cancelled</MenuItem>
              <MenuItem value="running">Running</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Sort By</InputLabel>
            <Select
              value={filters.sortBy}
              label="Sort By"
              onChange={(e: SelectChangeEvent) => handleFilterChange('sortBy', e.target.value)}
            >
              <MenuItem value="time">Time</MenuItem>
              <MenuItem value="duration">Duration</MenuItem>
              <MenuItem value="status">Status</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel>Order</InputLabel>
            <Select
              value={filters.sortOrder}
              label="Order"
              onChange={(e: SelectChangeEvent) => handleFilterChange('sortOrder', e.target.value)}
            >
              <MenuItem value="desc">Latest First</MenuItem>
              <MenuItem value="asc">Oldest First</MenuItem>
            </Select>
          </FormControl>
        </Stack>
      </Stack>

      <Divider sx={{ mb: 2 }} />

      {/* Results */}
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {filteredAndSortedHistory.length} results
      </Typography>

      <List sx={{ bgcolor: 'background.paper' }}>
        {filteredAndSortedHistory.map((execution) => (
          <ListItem
            key={execution.id}
            disablePadding
            selected={execution.id === selectedId}
            sx={{
              mb: 1,
              border: 1,
              borderColor: 'divider',
              borderRadius: 1,
              bgcolor: execution.id === selectedId ? 'action.selected' : 'transparent',
            }}
          >
            <ListItemButton onClick={() => onSelect(execution)}>
              <ListItemIcon>
                {getStatusIcon(execution.status)}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                      {JSON.stringify(execution.input).substring(0, 50)}...
                    </Typography>
                    <Chip
                      label={execution.status}
                      size="small"
                      color={
                        execution.status === 'success' ? 'success' :
                        execution.status === 'error' ? 'error' :
                        'default'
                      }
                    />
                  </Stack>
                }
                secondary={
                  <Stack direction="row" spacing={2} alignItems="center">
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      <AccessTime fontSize="small" />
                      <Typography variant="caption">
                        {formatTime(execution.startTime)}
                      </Typography>
                    </Stack>
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      <Speed fontSize="small" />
                      <Typography variant="caption">
                        {formatDuration(execution)}
                      </Typography>
                    </Stack>
                    {execution.output && (
                      <Stack direction="row" spacing={0.5} alignItems="center">
                        <DataObject fontSize="small" />
                        <Typography variant="caption">
                          {JSON.stringify(execution.output).length} chars
                        </Typography>
                      </Stack>
                    )}
                  </Stack>
                }
              />
              <ListItemSecondaryAction>
                <IconButton
                  edge="end"
                  onClick={(e) => handleMenuClick(e, execution)}
                >
                  <MoreVert />
                </IconButton>
              </ListItemSecondaryAction>
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => selectedExecution && handleCopy(selectedExecution)}>
          <ListItemIcon>
            <ContentCopy fontSize="small" />
          </ListItemIcon>
          <ListItemText>Copy as JSON</ListItemText>
        </MenuItem>
        {onDelete && (
          <MenuItem onClick={() => selectedExecution && handleDelete(selectedExecution)}>
            <ListItemIcon>
              <Delete fontSize="small" />
            </ListItemIcon>
            <ListItemText>Delete</ListItemText>
          </MenuItem>
        )}
      </Menu>
    </Box>
  );
};

export { ExecutionHistory };
export default ExecutionHistory;