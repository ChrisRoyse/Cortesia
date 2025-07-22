import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  IconButton,
  Chip,
  Box,
  Typography,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as ExecuteIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  CheckCircle as HealthyIcon,
  Warning as DegradedIcon,
  Error as UnavailableIcon,
  Help as UnknownIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { MCPTool, ToolStatus } from '../../types';

interface ToolTableProps {
  tools: MCPTool[];
  onToolClick: (tool: MCPTool) => void;
  onToolExecute: (tool: MCPTool) => void;
  onToggleFavorite: (toolId: string) => void;
  isFavorite: (toolId: string) => boolean;
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

export const ToolTable: React.FC<ToolTableProps> = ({
  tools,
  onToolClick,
  onToolExecute,
  onToggleFavorite,
  isFavorite,
}) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const paginatedTools = tools.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  return (
    <Paper>
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell width={40}></TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Category</TableCell>
              <TableCell>Status</TableCell>
              <TableCell align="right">Response Time</TableCell>
              <TableCell align="right">Success Rate</TableCell>
              <TableCell align="right">Executions</TableCell>
              <TableCell>Tags</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedTools.map((tool) => {
              const statusInfo = statusConfig[tool.status.health];
              const categoryColor = categoryColors[tool.category] || '#757575';
              const favorite = isFavorite(tool.id);

              return (
                <TableRow
                  key={tool.id}
                  hover
                  sx={{ cursor: 'pointer' }}
                  onClick={() => onToolClick(tool)}
                >
                  <TableCell onClick={(e) => e.stopPropagation()}>
                    <IconButton
                      size="small"
                      onClick={() => onToggleFavorite(tool.id)}
                    >
                      {favorite ? <StarIcon color="primary" fontSize="small" /> : <StarBorderIcon fontSize="small" />}
                    </IconButton>
                  </TableCell>
                  <TableCell>
                    <Box>
                      <Typography variant="body2" fontWeight="medium">
                        {tool.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        v{tool.version}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={tool.category}
                      size="small"
                      sx={{
                        backgroundColor: categoryColor,
                        color: 'white',
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      <Box sx={{ color: statusInfo.color }}>
                        {statusInfo.icon}
                      </Box>
                      <Typography variant="body2" sx={{ color: statusInfo.color }}>
                        {tool.status.health}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2">
                      {tool.metrics.averageResponseTime.toFixed(0)}ms
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography
                      variant="body2"
                      sx={{
                        color: tool.metrics.successRate > 95 ? '#4CAF50' : tool.metrics.successRate > 80 ? '#FF9800' : '#F44336',
                      }}
                    >
                      {tool.metrics.successRate.toFixed(1)}%
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2">
                      {tool.metrics.totalExecutions.toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box display="flex" gap={0.5} flexWrap="wrap">
                      {tool.tags?.slice(0, 2).map(tag => (
                        <Chip
                          key={tag}
                          label={tag}
                          size="small"
                          variant="outlined"
                          sx={{ fontSize: '0.7rem' }}
                        />
                      ))}
                      {tool.tags && tool.tags.length > 2 && (
                        <Tooltip title={tool.tags.slice(2).join(', ')}>
                          <Chip
                            label={`+${tool.tags.length - 2}`}
                            size="small"
                            variant="outlined"
                            sx={{ fontSize: '0.7rem' }}
                          />
                        </Tooltip>
                      )}
                    </Box>
                  </TableCell>
                  <TableCell align="center" onClick={(e) => e.stopPropagation()}>
                    <Box display="flex" justifyContent="center" gap={0.5}>
                      <Tooltip title="View details">
                        <IconButton
                          size="small"
                          onClick={() => onToolClick(tool)}
                        >
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Execute tool">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => onToolExecute(tool)}
                        >
                          <ExecuteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[5, 10, 25, 50]}
        component="div"
        count={tools.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Paper>
  );
};

export default ToolTable;