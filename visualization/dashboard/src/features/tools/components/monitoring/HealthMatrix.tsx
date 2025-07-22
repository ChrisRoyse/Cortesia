import React, { useState, useMemo } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Tooltip,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
  InputAdornment,
  IconButton,
  Fade,
  useTheme
} from '@mui/material';
import {
  GridView as GridViewIcon,
  ViewList as ViewListIcon,
  Search as SearchIcon,
  Clear as ClearIcon
} from '@mui/icons-material';
import { MCPTool, ToolCategory, ToolStatus } from '../../types';
import { MiniStatusIndicator, StatusProgressRing } from './StatusIndicator';

interface HealthMatrixProps {
  tools: MCPTool[];
  onToolClick?: (tool: MCPTool) => void;
  viewMode?: 'grid' | 'list';
  groupBy?: 'category' | 'status' | 'none';
}

interface ToolGroup {
  label: string;
  tools: MCPTool[];
  overallHealth: ToolStatus;
}

const HealthMatrix: React.FC<HealthMatrixProps> = ({
  tools,
  onToolClick,
  viewMode: initialViewMode = 'grid',
  groupBy: initialGroupBy = 'category'
}) => {
  const theme = useTheme();
  const [viewMode, setViewMode] = useState(initialViewMode);
  const [groupBy, setGroupBy] = useState(initialGroupBy);
  const [searchTerm, setSearchTerm] = useState('');
  const [hoveredTool, setHoveredTool] = useState<string | null>(null);

  // Filter tools based on search
  const filteredTools = useMemo(() => {
    if (!searchTerm) return tools;
    
    const search = searchTerm.toLowerCase();
    return tools.filter(tool => 
      tool.name.toLowerCase().includes(search) ||
      tool.description.toLowerCase().includes(search) ||
      tool.category.toLowerCase().includes(search) ||
      tool.tags?.some(tag => tag.toLowerCase().includes(search))
    );
  }, [tools, searchTerm]);

  // Group tools
  const groupedTools = useMemo((): ToolGroup[] => {
    if (groupBy === 'none') {
      return [{
        label: 'All Tools',
        tools: filteredTools,
        overallHealth: calculateOverallHealth(filteredTools)
      }];
    }

    const groups: Record<string, MCPTool[]> = {};

    filteredTools.forEach(tool => {
      const key = groupBy === 'category' ? tool.category : tool.status.health;
      if (!groups[key]) {
        groups[key] = [];
      }
      groups[key].push(tool);
    });

    // Sort groups
    const sortedKeys = Object.keys(groups).sort((a, b) => {
      if (groupBy === 'status') {
        const statusOrder: Record<ToolStatus, number> = {
          healthy: 0,
          degraded: 1,
          unavailable: 2,
          unknown: 3
        };
        return statusOrder[a as ToolStatus] - statusOrder[b as ToolStatus];
      }
      return a.localeCompare(b);
    });

    return sortedKeys.map(key => ({
      label: formatGroupLabel(key, groupBy),
      tools: groups[key],
      overallHealth: calculateOverallHealth(groups[key])
    }));
  }, [filteredTools, groupBy]);

  // Calculate overall health for a group
  function calculateOverallHealth(groupTools: MCPTool[]): ToolStatus {
    if (groupTools.length === 0) return 'unknown';
    
    const statusCounts = groupTools.reduce((acc, tool) => {
      acc[tool.status.health] = (acc[tool.status.health] || 0) + 1;
      return acc;
    }, {} as Record<ToolStatus, number>);

    if (statusCounts.unavailable > 0) return 'unavailable';
    if (statusCounts.degraded > 0) return 'degraded';
    if (statusCounts.unknown > 0) return 'unknown';
    return 'healthy';
  }

  // Format group label
  function formatGroupLabel(key: string, groupingType: string): string {
    if (groupingType === 'category') {
      return key.split('-').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ');
    }
    return key.charAt(0).toUpperCase() + key.slice(1);
  }

  // Get health color
  function getHealthColor(status: ToolStatus, opacity: number = 1): string {
    const colors = {
      healthy: theme.palette.success.main,
      degraded: theme.palette.warning.main,
      unavailable: theme.palette.error.main,
      unknown: theme.palette.grey[500]
    };
    
    const hex = colors[status];
    if (opacity === 1) return hex;
    
    // Convert hex to rgba
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
  }

  // Get status statistics
  const statusStats = useMemo(() => {
    const stats = {
      healthy: 0,
      degraded: 0,
      unavailable: 0,
      unknown: 0,
      total: filteredTools.length
    };

    filteredTools.forEach(tool => {
      stats[tool.status.health]++;
    });

    return stats;
  }, [filteredTools]);

  // Render tool cell (grid view)
  const ToolCell: React.FC<{ tool: MCPTool }> = ({ tool }) => {
    const isHovered = hoveredTool === tool.id;
    const cellSize = 80;
    
    return (
      <Tooltip
        title={
          <Box>
            <Typography variant="subtitle2">{tool.name}</Typography>
            <Typography variant="caption">
              {tool.category} - {tool.status.health}
            </Typography>
            <Typography variant="caption" display="block">
              Response: {tool.status.responseTime}ms
            </Typography>
            <Typography variant="caption" display="block">
              Error Rate: {(tool.status.errorRate * 100).toFixed(1)}%
            </Typography>
          </Box>
        }
        arrow
        placement="top"
      >
        <Paper
          elevation={isHovered ? 4 : 1}
          sx={{
            width: cellSize,
            height: cellSize,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            backgroundColor: getHealthColor(tool.status.health, 0.1),
            borderLeft: `4px solid ${getHealthColor(tool.status.health)}`,
            transition: 'all 0.2s ease',
            transform: isHovered ? 'scale(1.05)' : 'scale(1)',
            '&:hover': {
              backgroundColor: getHealthColor(tool.status.health, 0.2)
            }
          }}
          onClick={() => onToolClick?.(tool)}
          onMouseEnter={() => setHoveredTool(tool.id)}
          onMouseLeave={() => setHoveredTool(null)}
        >
          <StatusProgressRing status={tool.status} size={36} thickness={3} />
          <Typography
            variant="caption"
            sx={{
              mt: 1,
              textAlign: 'center',
              fontSize: '0.65rem',
              lineHeight: 1.2,
              maxWidth: cellSize - 8,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical'
            }}
          >
            {tool.name}
          </Typography>
        </Paper>
      </Tooltip>
    );
  };

  // Render tool row (list view)
  const ToolRow: React.FC<{ tool: MCPTool }> = ({ tool }) => {
    const isHovered = hoveredTool === tool.id;
    
    return (
      <Paper
        elevation={isHovered ? 2 : 0}
        sx={{
          p: 1.5,
          mb: 1,
          display: 'flex',
          alignItems: 'center',
          cursor: 'pointer',
          backgroundColor: isHovered ? 'action.hover' : 'transparent',
          transition: 'all 0.2s ease',
          '&:hover': {
            backgroundColor: 'action.hover'
          }
        }}
        onClick={() => onToolClick?.(tool)}
        onMouseEnter={() => setHoveredTool(tool.id)}
        onMouseLeave={() => setHoveredTool(null)}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
          <MiniStatusIndicator status={tool.status.health} />
          <Typography variant="body2" sx={{ ml: 2, flex: 1 }}>
            {tool.name}
          </Typography>
          <Chip
            label={tool.category}
            size="small"
            variant="outlined"
            sx={{ mx: 2 }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mr: 2 }}>
            {tool.status.responseTime}ms
          </Typography>
          <StatusProgressRing status={tool.status} size={30} thickness={3} />
        </Box>
      </Paper>
    );
  };

  return (
    <Box>
      {/* Controls */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        {/* Search */}
        <TextField
          size="small"
          placeholder="Search tools..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          sx={{ minWidth: 250 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" />
              </InputAdornment>
            ),
            endAdornment: searchTerm && (
              <InputAdornment position="end">
                <IconButton size="small" onClick={() => setSearchTerm('')}>
                  <ClearIcon fontSize="small" />
                </IconButton>
              </InputAdornment>
            )
          }}
        />

        {/* View Mode */}
        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={(_, value) => value && setViewMode(value)}
          size="small"
        >
          <ToggleButton value="grid">
            <Tooltip title="Grid View">
              <GridViewIcon fontSize="small" />
            </Tooltip>
          </ToggleButton>
          <ToggleButton value="list">
            <Tooltip title="List View">
              <ViewListIcon fontSize="small" />
            </Tooltip>
          </ToggleButton>
        </ToggleButtonGroup>

        {/* Group By */}
        <ToggleButtonGroup
          value={groupBy}
          exclusive
          onChange={(_, value) => value && setGroupBy(value)}
          size="small"
        >
          <ToggleButton value="category">Category</ToggleButton>
          <ToggleButton value="status">Status</ToggleButton>
          <ToggleButton value="none">None</ToggleButton>
        </ToggleButtonGroup>

        {/* Status Summary */}
        <Box sx={{ display: 'flex', gap: 1, ml: 'auto' }}>
          <Chip
            icon={<MiniStatusIndicator status="healthy" />}
            label={statusStats.healthy}
            size="small"
            color="success"
            variant="outlined"
          />
          <Chip
            icon={<MiniStatusIndicator status="degraded" />}
            label={statusStats.degraded}
            size="small"
            color="warning"
            variant="outlined"
          />
          <Chip
            icon={<MiniStatusIndicator status="unavailable" />}
            label={statusStats.unavailable}
            size="small"
            color="error"
            variant="outlined"
          />
        </Box>
      </Box>

      {/* Tool Matrix */}
      {groupedTools.map(group => (
        <Fade in key={group.label}>
          <Box sx={{ mb: 3 }}>
            {/* Group Header */}
            {groupBy !== 'none' && (
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ flex: 1 }}>
                  {group.label}
                </Typography>
                <Chip
                  label={`${group.tools.length} tools`}
                  size="small"
                  sx={{ mr: 1 }}
                />
                <MiniStatusIndicator status={group.overallHealth} />
              </Box>
            )}

            {/* Tools Grid/List */}
            {viewMode === 'grid' ? (
              <Grid container spacing={1}>
                {group.tools.map(tool => (
                  <Grid item key={tool.id}>
                    <ToolCell tool={tool} />
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Box>
                {group.tools.map(tool => (
                  <ToolRow key={tool.id} tool={tool} />
                ))}
              </Box>
            )}
          </Box>
        </Fade>
      ))}

      {/* Empty State */}
      {filteredTools.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="body1" color="text.secondary">
            {searchTerm ? 'No tools match your search' : 'No tools available'}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default HealthMatrix;