import React from 'react';
import {
  Paper,
  Box,
  Typography,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Chip,
  Button,
  Divider,
  Stack,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../../../stores/hooks';
import {
  toggleFilterCategory,
  toggleFilterStatus,
  toggleFilterTag,
  clearFilters,
} from '../../stores/toolsSlice';
import { ToolCategory, ToolStatus } from '../../types';

interface FilterPanelProps {
  categories: ToolCategory[];
  onClose?: () => void;
  sx?: any;
}

const categoryInfo: Record<ToolCategory, { label: string; color: string }> = {
  'knowledge-graph': { label: 'Knowledge Graph', color: '#4CAF50' },
  'cognitive': { label: 'Cognitive', color: '#2196F3' },
  'neural': { label: 'Neural', color: '#FF9800' },
  'memory': { label: 'Memory', color: '#9C27B0' },
  'analysis': { label: 'Analysis', color: '#F44336' },
  'federation': { label: 'Federation', color: '#00BCD4' },
  'utility': { label: 'Utility', color: '#607D8B' },
};

const statusInfo: Record<ToolStatus, { label: string; color: string }> = {
  healthy: { label: 'Healthy', color: '#4CAF50' },
  degraded: { label: 'Degraded', color: '#FF9800' },
  unavailable: { label: 'Unavailable', color: '#F44336' },
  unknown: { label: 'Unknown', color: '#9E9E9E' },
};

export const FilterPanel: React.FC<FilterPanelProps> = ({
  categories,
  onClose,
  sx,
}) => {
  const dispatch = useAppDispatch();
  const { filters, tools } = useAppSelector(state => state.tools);
  const [expandedSections, setExpandedSections] = React.useState({
    categories: true,
    status: true,
    tags: false,
  });

  // Get all unique tags from tools
  const allTags = React.useMemo(() => {
    const tagSet = new Set<string>();
    tools.forEach(tool => {
      tool.tags?.forEach(tag => tagSet.add(tag));
    });
    return Array.from(tagSet).sort();
  }, [tools]);

  // Count tools per category
  const categoryCounts = React.useMemo(() => {
    const counts: Record<string, number> = {};
    tools.forEach(tool => {
      counts[tool.category] = (counts[tool.category] || 0) + 1;
    });
    return counts;
  }, [tools]);

  // Count tools per status
  const statusCounts = React.useMemo(() => {
    const counts: Record<string, number> = {};
    tools.forEach(tool => {
      counts[tool.status.health] = (counts[tool.status.health] || 0) + 1;
    });
    return counts;
  }, [tools]);

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const activeFilterCount = filters.categories.length + filters.status.length + filters.tags.length;

  return (
    <Paper sx={{ p: 2, ...sx }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">Filters</Typography>
        <Box display="flex" alignItems="center" gap={1}>
          {activeFilterCount > 0 && (
            <>
              <Chip
                label={`${activeFilterCount} active`}
                size="small"
                color="primary"
              />
              <Button
                size="small"
                startIcon={<ClearIcon />}
                onClick={() => dispatch(clearFilters())}
              >
                Clear all
              </Button>
            </>
          )}
          {onClose && (
            <IconButton size="small" onClick={onClose}>
              <ClearIcon />
            </IconButton>
          )}
        </Box>
      </Box>

      <Stack spacing={2}>
        {/* Categories */}
        <Box>
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            sx={{ cursor: 'pointer' }}
            onClick={() => toggleSection('categories')}
          >
            <Typography variant="subtitle2">Categories</Typography>
            <IconButton size="small">
              {expandedSections.categories ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
          <Collapse in={expandedSections.categories}>
            <FormGroup sx={{ mt: 1 }}>
              {categories.map(category => (
                <FormControlLabel
                  key={category}
                  control={
                    <Checkbox
                      checked={filters.categories.includes(category)}
                      onChange={() => dispatch(toggleFilterCategory(category))}
                      size="small"
                    />
                  }
                  label={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Chip
                        label={categoryInfo[category].label}
                        size="small"
                        sx={{
                          backgroundColor: categoryInfo[category].color,
                          color: 'white',
                        }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        ({categoryCounts[category] || 0})
                      </Typography>
                    </Box>
                  }
                />
              ))}
            </FormGroup>
          </Collapse>
        </Box>

        <Divider />

        {/* Status */}
        <Box>
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            sx={{ cursor: 'pointer' }}
            onClick={() => toggleSection('status')}
          >
            <Typography variant="subtitle2">Status</Typography>
            <IconButton size="small">
              {expandedSections.status ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
          <Collapse in={expandedSections.status}>
            <FormGroup sx={{ mt: 1 }}>
              {(Object.keys(statusInfo) as ToolStatus[]).map(status => (
                <FormControlLabel
                  key={status}
                  control={
                    <Checkbox
                      checked={filters.status.includes(status)}
                      onChange={() => dispatch(toggleFilterStatus(status))}
                      size="small"
                    />
                  }
                  label={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography
                        variant="body2"
                        sx={{ color: statusInfo[status].color }}
                      >
                        {statusInfo[status].label}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        ({statusCounts[status] || 0})
                      </Typography>
                    </Box>
                  }
                />
              ))}
            </FormGroup>
          </Collapse>
        </Box>

        {allTags.length > 0 && (
          <>
            <Divider />

            {/* Tags */}
            <Box>
              <Box
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                sx={{ cursor: 'pointer' }}
                onClick={() => toggleSection('tags')}
              >
                <Typography variant="subtitle2">Tags</Typography>
                <IconButton size="small">
                  {expandedSections.tags ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </IconButton>
              </Box>
              <Collapse in={expandedSections.tags}>
                <Box sx={{ mt: 1, maxHeight: 200, overflow: 'auto' }}>
                  <FormGroup>
                    {allTags.map(tag => (
                      <FormControlLabel
                        key={tag}
                        control={
                          <Checkbox
                            checked={filters.tags.includes(tag)}
                            onChange={() => dispatch(toggleFilterTag(tag))}
                            size="small"
                          />
                        }
                        label={
                          <Typography variant="body2">
                            {tag}
                          </Typography>
                        }
                      />
                    ))}
                  </FormGroup>
                </Box>
              </Collapse>
            </Box>
          </>
        )}
      </Stack>
    </Paper>
  );
};

export default FilterPanel;