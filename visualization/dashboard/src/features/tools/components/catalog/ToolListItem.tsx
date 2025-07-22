import React from 'react';
import {
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Chip,
  Box,
  Typography,
  Tooltip,
  Stack,
} from '@mui/material';
import {
  PlayArrow as ExecuteIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  CheckCircle as HealthyIcon,
  Warning as DegradedIcon,
  Error as UnavailableIcon,
  Help as UnknownIcon,
  Category as CategoryIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { MCPTool, ToolStatus } from '../../types';

interface ToolListItemProps {
  tool: MCPTool;
  onClick: () => void;
  onExecute: () => void;
  onToggleFavorite: () => void;
  isFavorite: boolean;
}

const statusIcons: Record<ToolStatus, React.ReactNode> = {
  healthy: <HealthyIcon color="success" fontSize="small" />,
  degraded: <DegradedIcon color="warning" fontSize="small" />,
  unavailable: <UnavailableIcon color="error" fontSize="small" />,
  unknown: <UnknownIcon color="disabled" fontSize="small" />,
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

export const ToolListItem: React.FC<ToolListItemProps> = ({
  tool,
  onClick,
  onExecute,
  onToggleFavorite,
  isFavorite,
}) => {
  const categoryColor = categoryColors[tool.category] || '#757575';

  return (
    <ListItem
      disablePadding
      secondaryAction={
        <Stack direction="row" spacing={1} alignItems="center">
          {/* Metrics */}
          <Box display="flex" alignItems="center" gap={2}>
            <Tooltip title="Average response time">
              <Box display="flex" alignItems="center" gap={0.5}>
                <SpeedIcon fontSize="small" color="action" />
                <Typography variant="caption" color="text.secondary">
                  {tool.metrics.averageResponseTime.toFixed(0)}ms
                </Typography>
              </Box>
            </Tooltip>
            <Tooltip title="Success rate">
              <Typography
                variant="caption"
                sx={{
                  color: tool.metrics.successRate > 95 ? '#4CAF50' : tool.metrics.successRate > 80 ? '#FF9800' : '#F44336',
                }}
              >
                {tool.metrics.successRate.toFixed(1)}%
              </Typography>
            </Tooltip>
          </Box>

          {/* Actions */}
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
        </Stack>
      }
    >
      <ListItemButton onClick={onClick}>
        <ListItemIcon>
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: 1,
              backgroundColor: categoryColor + '20',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <CategoryIcon sx={{ color: categoryColor }} />
          </Box>
        </ListItemIcon>
        <ListItemText
          primary={
            <Box display="flex" alignItems="center" gap={1}>
              <Typography variant="body1" fontWeight="medium">
                {tool.name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                v{tool.version}
              </Typography>
              {statusIcons[tool.status.health]}
              <Chip
                label={tool.category}
                size="small"
                sx={{
                  backgroundColor: categoryColor,
                  color: 'white',
                  fontSize: '0.7rem',
                  height: 20,
                }}
              />
              {tool.tags && tool.tags.slice(0, 2).map(tag => (
                <Chip
                  key={tag}
                  label={tag}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.7rem', height: 20 }}
                />
              ))}
            </Box>
          }
          secondary={
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {tool.description}
            </Typography>
          }
        />
      </ListItemButton>
    </ListItem>
  );
};

export default ToolListItem;