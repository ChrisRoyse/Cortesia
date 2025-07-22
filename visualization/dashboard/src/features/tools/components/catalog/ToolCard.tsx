import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Chip,
  Box,
  IconButton,
  Tooltip,
  Stack,
  LinearProgress,
} from '@mui/material';
import {
  PlayArrow as ExecuteIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  CheckCircle as HealthyIcon,
  Warning as DegradedIcon,
  Error as UnavailableIcon,
  Help as UnknownIcon,
  Speed as SpeedIcon,
  Functions as FunctionsIcon,
} from '@mui/icons-material';
import { MCPTool, ToolStatus } from '../../types';

interface ToolCardProps {
  tool: MCPTool;
  onClick: () => void;
  onExecute: () => void;
  onToggleFavorite: () => void;
  isFavorite: boolean;
  compact?: boolean;
}

const statusConfig: Record<ToolStatus, { icon: React.ReactNode; color: string }> = {
  healthy: { icon: <HealthyIcon fontSize="small" />, color: '#4CAF50' },
  degraded: { icon: <DegradedIcon fontSize="small" />, color: '#FF9800' },
  unavailable: { icon: <UnavailableIcon fontSize="small" />, color: '#F44336' },
  unknown: { icon: <UnknownIcon fontSize="small" />, color: '#9E9E9E' },
};

const categoryColors: Record<string, string> = {
  'knowledge-graph': '#4CAF50',
  'cognitive': '#2196F3',
  'neural': '#FF9800',
  'memory': '#9C27B0',
  'analysis': '#F44336',
  'federation': '#00BCD4',
  'utility': '#607D8B',
};

export const ToolCard: React.FC<ToolCardProps> = ({
  tool,
  onClick,
  onExecute,
  onToggleFavorite,
  isFavorite,
  compact = false,
}) => {
  const statusInfo = statusConfig[tool.status.health];
  const categoryColor = categoryColors[tool.category] || '#757575';

  // Calculate performance score (0-100)
  const performanceScore = Math.max(0, Math.min(100, 
    100 - (tool.metrics.averageResponseTime / 1000) * 20
  ));

  return (
    <Card
      sx={{
        height: compact ? 'auto' : '100%',
        display: 'flex',
        flexDirection: 'column',
        cursor: 'pointer',
        transition: 'all 0.2s',
        '&:hover': {
          boxShadow: 4,
          transform: 'translateY(-2px)',
        },
      }}
      onClick={onClick}
    >
      <CardContent sx={{ flexGrow: 1, pb: compact ? 1 : 2 }}>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
          <Box flexGrow={1}>
            <Typography
              variant={compact ? 'body1' : 'h6'}
              component="div"
              noWrap
              fontWeight="medium"
            >
              {tool.name}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              v{tool.version}
            </Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={0.5}>
            <Tooltip title={`Status: ${tool.status.health}`}>
              <Box sx={{ color: statusInfo.color }}>
                {statusInfo.icon}
              </Box>
            </Tooltip>
            <Tooltip title={isFavorite ? 'Remove from favorites' : 'Add to favorites'}>
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleFavorite();
                }}
              >
                {isFavorite ? <StarIcon color="primary" fontSize="small" /> : <StarBorderIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Category chip */}
        <Chip
          label={tool.category}
          size="small"
          sx={{
            backgroundColor: categoryColor,
            color: 'white',
            mb: 1,
          }}
        />

        {/* Description */}
        {!compact && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              mb: 2,
            }}
          >
            {tool.description}
          </Typography>
        )}

        {/* Metrics */}
        <Stack spacing={1}>
          {/* Performance */}
          <Box>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="caption" color="text.secondary">
                Performance
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {tool.metrics.averageResponseTime.toFixed(0)}ms
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={performanceScore}
              sx={{
                height: 4,
                borderRadius: 2,
                backgroundColor: 'action.hover',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: performanceScore > 80 ? '#4CAF50' : performanceScore > 50 ? '#FF9800' : '#F44336',
                },
              }}
            />
          </Box>

          {/* Success rate */}
          {!compact && (
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="caption" color="text.secondary">
                Success Rate
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: tool.metrics.successRate > 95 ? '#4CAF50' : tool.metrics.successRate > 80 ? '#FF9800' : '#F44336',
                }}
              >
                {tool.metrics.successRate.toFixed(1)}%
              </Typography>
            </Box>
          )}

          {/* Executions */}
          {!compact && tool.metrics.totalExecutions > 0 && (
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="caption" color="text.secondary">
                Executions
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {tool.metrics.totalExecutions.toLocaleString()}
              </Typography>
            </Box>
          )}
        </Stack>

        {/* Tags */}
        {!compact && tool.tags && tool.tags.length > 0 && (
          <Box mt={1.5} display="flex" gap={0.5} flexWrap="wrap">
            {tool.tags.slice(0, 3).map(tag => (
              <Chip
                key={tag}
                label={tag}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
            ))}
            {tool.tags.length > 3 && (
              <Chip
                label={`+${tool.tags.length - 3}`}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
            )}
          </Box>
        )}
      </CardContent>

      <CardActions sx={{ justifyContent: 'flex-end', pt: 0 }}>
        <Tooltip title="Execute tool">
          <IconButton
            size="small"
            color="primary"
            onClick={(e) => {
              e.stopPropagation();
              onExecute();
            }}
          >
            <ExecuteIcon />
          </IconButton>
        </Tooltip>
      </CardActions>
    </Card>
  );
};

export default ToolCard;