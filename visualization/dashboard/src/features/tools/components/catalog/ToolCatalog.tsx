import React, { useState, useCallback, useMemo } from 'react';
import {
  Grid,
  Box,
  Paper,
  Typography,
  TextField,
  InputAdornment,
  IconButton,
  Chip,
  Card,
  CardContent,
  CardActions,
  Button,
  Menu,
  MenuItem,
  Tooltip,
  CircularProgress,
  Alert,
  AlertTitle,
  Tabs,
  Tab,
  Badge,
  Stack,
  Divider,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  ViewModule as GridViewIcon,
  ViewList as ListViewIcon,
  TableChart as TableViewIcon,
  Refresh as RefreshIcon,
  Sort as SortIcon,
  MoreVert as MoreIcon,
  PlayArrow as ExecuteIcon,
  Info as InfoIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  CheckCircle as HealthyIcon,
  Warning as DegradedIcon,
  Error as UnavailableIcon,
  Help as UnknownIcon,
  Category as CategoryIcon,
  Speed as PerformanceIcon,
  AccessTime as LastUsedIcon,
} from '@mui/icons-material';
import { useToolDiscovery } from '../../hooks/useToolDiscovery';
import { useToolRegistry, useToolFavorites } from '../../hooks/useToolRegistry';
import { MCPTool, ToolCategory, ToolStatus } from '../../types';
import { useAppDispatch, useAppSelector } from '../../../../stores/hooks';
import {
  setView,
  setSortBy,
  toggleSortOrder,
  setSearchTerm,
  toggleFilterCategory,
  toggleFilterStatus,
  clearFilters,
  selectFilteredTools,
} from '../../stores/toolsSlice';
import ToolCard from './ToolCard';
import ToolListItem from './ToolListItem';
import ToolTable from './ToolTable';
import ToolDetailsDialog from './ToolDetailsDialog';
import ToolExecutionDialog from './ToolExecutionDialog';
import FilterPanel from './FilterPanel';

interface ToolCatalogProps {
  onToolSelect?: (tool: MCPTool) => void;
  onToolExecute?: (tool: MCPTool) => void;
  initialView?: 'grid' | 'list' | 'table';
  showFavorites?: boolean;
  compactMode?: boolean;
}

const categoryInfo: Record<ToolCategory, { label: string; color: string; icon: React.ReactNode }> = {
  'knowledge-graph': { label: 'Knowledge Graph', color: '#4CAF50', icon: <CategoryIcon /> },
  'cognitive': { label: 'Cognitive', color: '#2196F3', icon: <CategoryIcon /> },
  'neural': { label: 'Neural', color: '#FF9800', icon: <CategoryIcon /> },
  'memory': { label: 'Memory', color: '#9C27B0', icon: <CategoryIcon /> },
  'analysis': { label: 'Analysis', color: '#F44336', icon: <CategoryIcon /> },
  'federation': { label: 'Federation', color: '#00BCD4', icon: <CategoryIcon /> },
  'utility': { label: 'Utility', color: '#607D8B', icon: <CategoryIcon /> },
};

const statusIcons: Record<ToolStatus, React.ReactNode> = {
  healthy: <HealthyIcon color="success" />,
  degraded: <DegradedIcon color="warning" />,
  unavailable: <UnavailableIcon color="error" />,
  unknown: <UnknownIcon color="disabled" />,
};

export const ToolCatalog: React.FC<ToolCatalogProps> = ({
  onToolSelect,
  onToolExecute,
  initialView = 'grid',
  showFavorites = true,
  compactMode = false,
}) => {
  const dispatch = useAppDispatch();
  const { view, sortBy, sortOrder, filters } = useAppSelector(state => state.tools);
  const filteredTools = useAppSelector(selectFilteredTools);
  
  const { tools, isDiscovering, error, discoverTools, discoveryStats, lastDiscovery } = useToolDiscovery();
  const { categories, registryStats } = useToolRegistry();
  const { favorites, toggleFavorite, isFavorite } = useToolFavorites();

  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [executionOpen, setExecutionOpen] = useState(false);
  const [filterMenuAnchor, setFilterMenuAnchor] = useState<null | HTMLElement>(null);
  const [sortMenuAnchor, setSortMenuAnchor] = useState<null | HTMLElement>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0);

  // Handle tool selection
  const handleToolClick = useCallback((tool: MCPTool) => {
    setSelectedTool(tool);
    setDetailsOpen(true);
    onToolSelect?.(tool);
  }, [onToolSelect]);

  // Handle tool execution
  const handleToolExecute = useCallback((tool: MCPTool) => {
    setSelectedTool(tool);
    setExecutionOpen(true);
    onToolExecute?.(tool);
  }, [onToolExecute]);

  // Handle view change
  const handleViewChange = useCallback((newView: 'grid' | 'list' | 'table') => {
    dispatch(setView(newView));
  }, [dispatch]);

  // Handle sort change
  const handleSortChange = useCallback((event: SelectChangeEvent) => {
    dispatch(setSortBy(event.target.value as any));
  }, [dispatch]);

  // Handle search
  const handleSearchChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    dispatch(setSearchTerm(event.target.value));
  }, [dispatch]);

  // Get tools to display based on selected tab
  const displayTools = useMemo(() => {
    if (selectedTab === 1 && showFavorites) {
      // Favorites tab
      return filteredTools.filter(tool => isFavorite(tool.id));
    }
    return filteredTools;
  }, [filteredTools, selectedTab, showFavorites, isFavorite]);

  // Group tools by category
  const groupedTools = useMemo(() => {
    const grouped: Record<ToolCategory, MCPTool[]> = {} as any;
    displayTools.forEach(tool => {
      if (!grouped[tool.category]) {
        grouped[tool.category] = [];
      }
      grouped[tool.category].push(tool);
    });
    return grouped;
  }, [displayTools]);

  // Render tool grid
  const renderToolGrid = () => (
    <Grid container spacing={2}>
      {Object.entries(groupedTools).map(([category, categoryTools]) => (
        categoryTools.length > 0 && (
          <React.Fragment key={category}>
            <Grid item xs={12}>
              <Box display="flex" alignItems="center" gap={1} mt={2} mb={1}>
                {categoryInfo[category as ToolCategory].icon}
                <Typography variant="h6" color="text.secondary">
                  {categoryInfo[category as ToolCategory].label}
                </Typography>
                <Chip
                  size="small"
                  label={categoryTools.length}
                  sx={{ backgroundColor: categoryInfo[category as ToolCategory].color, color: 'white' }}
                />
              </Box>
            </Grid>
            {categoryTools.map(tool => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={tool.id}>
                <ToolCard
                  tool={tool}
                  onClick={() => handleToolClick(tool)}
                  onExecute={() => handleToolExecute(tool)}
                  onToggleFavorite={() => toggleFavorite(tool.id)}
                  isFavorite={isFavorite(tool.id)}
                  compact={compactMode}
                />
              </Grid>
            ))}
          </React.Fragment>
        )
      ))}
    </Grid>
  );

  // Render tool list
  const renderToolList = () => (
    <Stack spacing={1}>
      {displayTools.map(tool => (
        <ToolListItem
          key={tool.id}
          tool={tool}
          onClick={() => handleToolClick(tool)}
          onExecute={() => handleToolExecute(tool)}
          onToggleFavorite={() => toggleFavorite(tool.id)}
          isFavorite={isFavorite(tool.id)}
        />
      ))}
    </Stack>
  );

  // Render tool table
  const renderToolTable = () => (
    <ToolTable
      tools={displayTools}
      onToolClick={handleToolClick}
      onToolExecute={handleToolExecute}
      onToggleFavorite={toggleFavorite}
      isFavorite={isFavorite}
    />
  );

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Stack spacing={2}>
          {/* Title and stats */}
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h5">MCP Tool Catalog</Typography>
            <Stack direction="row" spacing={2} alignItems="center">
              <Chip
                label={`${displayTools.length} / ${tools.length} tools`}
                color="primary"
                variant="outlined"
              />
              {lastDiscovery && (
                <Typography variant="caption" color="text.secondary">
                  Last updated: {new Date(lastDiscovery).toLocaleTimeString()}
                </Typography>
              )}
              <Tooltip title="Refresh tools">
                <IconButton onClick={() => discoverTools()} disabled={isDiscovering}>
                  {isDiscovering ? <CircularProgress size={20} /> : <RefreshIcon />}
                </IconButton>
              </Tooltip>
            </Stack>
          </Box>

          {/* Tabs */}
          {showFavorites && (
            <Tabs value={selectedTab} onChange={(_, v) => setSelectedTab(v)}>
              <Tab label="All Tools" />
              <Tab
                label={
                  <Badge badgeContent={favorites.length} color="primary">
                    Favorites
                  </Badge>
                }
              />
            </Tabs>
          )}

          {/* Search and filters */}
          <Box display="flex" gap={2} alignItems="center">
            <TextField
              fullWidth
              size="small"
              placeholder="Search tools..."
              value={filters.searchTerm}
              onChange={handleSearchChange}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />

            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Sort by</InputLabel>
              <Select value={sortBy} onChange={handleSortChange}>
                <MenuItem value="name">Name</MenuItem>
                <MenuItem value="category">Category</MenuItem>
                <MenuItem value="status">Status</MenuItem>
                <MenuItem value="performance">Performance</MenuItem>
                <MenuItem value="lastUsed">Last Used</MenuItem>
              </Select>
            </FormControl>

            <Tooltip title="Toggle sort order">
              <IconButton onClick={() => dispatch(toggleSortOrder())}>
                <SortIcon style={{ transform: sortOrder === 'desc' ? 'rotate(180deg)' : 'none' }} />
              </IconButton>
            </Tooltip>

            <Tooltip title="Filter tools">
              <IconButton onClick={() => setShowFilters(!showFilters)}>
                <Badge
                  badgeContent={filters.categories.length + filters.status.length + filters.tags.length}
                  color="primary"
                >
                  <FilterIcon />
                </Badge>
              </IconButton>
            </Tooltip>

            {/* View toggle */}
            <Box display="flex" border={1} borderColor="divider" borderRadius={1}>
              <Tooltip title="Grid view">
                <IconButton size="small" onClick={() => handleViewChange('grid')} color={view === 'grid' ? 'primary' : 'default'}>
                  <GridViewIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="List view">
                <IconButton size="small" onClick={() => handleViewChange('list')} color={view === 'list' ? 'primary' : 'default'}>
                  <ListViewIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Table view">
                <IconButton size="small" onClick={() => handleViewChange('table')} color={view === 'table' ? 'primary' : 'default'}>
                  <TableViewIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {/* Active filters */}
          {(filters.categories.length > 0 || filters.status.length > 0 || filters.tags.length > 0) && (
            <Box display="flex" gap={1} alignItems="center" flexWrap="wrap">
              <Typography variant="caption" color="text.secondary">Active filters:</Typography>
              {filters.categories.map(cat => (
                <Chip
                  key={cat}
                  size="small"
                  label={categoryInfo[cat as ToolCategory].label}
                  onDelete={() => dispatch(toggleFilterCategory(cat))}
                  sx={{ backgroundColor: categoryInfo[cat as ToolCategory].color, color: 'white' }}
                />
              ))}
              {filters.status.map(status => (
                <Chip
                  key={status}
                  size="small"
                  label={status}
                  onDelete={() => dispatch(toggleFilterStatus(status))}
                  icon={statusIcons[status as ToolStatus] as any}
                />
              ))}
              <Button size="small" onClick={() => dispatch(clearFilters())}>
                Clear all
              </Button>
            </Box>
          )}
        </Stack>
      </Paper>

      {/* Filter panel */}
      {showFilters && (
        <FilterPanel
          categories={categories}
          onClose={() => setShowFilters(false)}
          sx={{ mb: 2 }}
        />
      )}

      {/* Error display */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <AlertTitle>Discovery Error</AlertTitle>
          {error}
        </Alert>
      )}

      {/* Tool display */}
      <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
        {displayTools.length === 0 ? (
          <Box display="flex" justifyContent="center" alignItems="center" height="100%">
            <Typography variant="h6" color="text.secondary">
              {filters.searchTerm || filters.categories.length > 0 || filters.status.length > 0
                ? 'No tools match your filters'
                : 'No tools discovered yet'}
            </Typography>
          </Box>
        ) : (
          <>
            {view === 'grid' && renderToolGrid()}
            {view === 'list' && renderToolList()}
            {view === 'table' && renderToolTable()}
          </>
        )}
      </Box>

      {/* Dialogs */}
      {selectedTool && (
        <>
          <ToolDetailsDialog
            open={detailsOpen}
            tool={selectedTool}
            onClose={() => setDetailsOpen(false)}
            onExecute={() => {
              setDetailsOpen(false);
              setExecutionOpen(true);
            }}
          />
          <ToolExecutionDialog
            open={executionOpen}
            tool={selectedTool}
            onClose={() => setExecutionOpen(false)}
          />
        </>
      )}
    </Box>
  );
};

export default ToolCatalog;