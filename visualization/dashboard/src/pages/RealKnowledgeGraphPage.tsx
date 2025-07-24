import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Card,
  CardContent,
  CardHeader,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  TextField,
  Button,
  Tabs,
  Tab,
  IconButton,
  Collapse,
  Alert,
  CircularProgress,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Badge
} from '@mui/material';
import {
  Refresh,
  Search,
  ExpandMore,
  Storage,
  AccountTree,
  DataObject,
  Article,
  Analytics,
  Visibility,
  ExpandLess,
  Launch,
  FilterList,
  Wifi,
  WifiOff,
  Error as ErrorIcon
} from '@mui/icons-material';
import {
  RealEntity,
  RealTriple,
  RealRelationship,
  RealChunk,
  DatabaseInfo
} from '../services/KnowledgeGraphDataService';
import { useRealKnowledgeGraph, useEntityOperations } from '../hooks/useRealKnowledgeGraph';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`kg-tabpanel-${index}`}
      aria-labelledby={`kg-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const RealKnowledgeGraphPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedAccordions, setExpandedAccordions] = useState<Record<string, boolean>>({});

  // Use the real knowledge graph hook with auto-refresh and WebSocket
  const {
    data,
    loading,
    error,
    refreshData,
    connectionStatus,
    lastUpdated
  } = useRealKnowledgeGraph({
    autoRefresh: true,
    refreshInterval: 30000, // 30 seconds
    enableWebSocket: true
  });

  // Use entity operations hook
  const {
    selectedEntity,
    entityDetails,
    loadingDetails,
    selectEntity,
    clearSelection
  } = useEntityOperations();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleEntitySelect = (entity: RealEntity) => {
    selectEntity(entity);
    setTabValue(5); // Switch to entity details tab
  };

  const toggleAccordion = (panel: string) => {
    setExpandedAccordions(prev => ({
      ...prev,
      [panel]: !prev[panel]
    }));
  };

  // Filter data based on search term
  const filteredEntities = data?.entities.filter(entity =>
    entity.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    entity.entity_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
    entity.description.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  const filteredTriples = data?.triples.filter(triple =>
    triple.subject.toLowerCase().includes(searchTerm.toLowerCase()) ||
    triple.predicate.toLowerCase().includes(searchTerm.toLowerCase()) ||
    triple.object.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <Box sx={{ textAlign: 'center' }}>
          <CircularProgress size={60} />
          <Typography variant="h6" sx={{ mt: 2 }}>
            Loading Complete Knowledge Graph Data...
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Fetching entities, triples, relationships, and database information
          </Typography>
        </Box>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" action={
          <Button onClick={refreshData} startIcon={<Refresh />}>
            Retry
          </Button>
        }>
          <strong>Error loading knowledge graph data:</strong> {error}
        </Alert>
      </Box>
    );
  }

  if (!data) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="warning">
          No knowledge graph data available. Make sure the LLMKG backend is running.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="h4" gutterBottom>
                Real Knowledge Graph Explorer
              </Typography>
              {/* Connection Status Indicator */}
              <Tooltip title={`Connection: ${connectionStatus}`}>
                <IconButton size="small" color={connectionStatus === 'connected' ? 'success' : 'error'}>
                  {connectionStatus === 'connected' && <Wifi />}
                  {connectionStatus === 'disconnected' && <WifiOff />}
                  {connectionStatus === 'connecting' && <CircularProgress size={20} />}
                  {connectionStatus === 'error' && <ErrorIcon />}
                </IconButton>
              </Tooltip>
            </Box>
            <Typography variant="body1" color="text.secondary">
              Complete view of ALL data in the LLMKG knowledge graph system
              {lastUpdated && (
                <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 2 }}>
                  Last updated: {new Date(lastUpdated).toLocaleTimeString()}
                </Typography>
              )}
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <TextField
              size="small"
              placeholder="Search entities, triples, relationships..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />
              }}
              sx={{ minWidth: 300 }}
            />
            <Tooltip title="Refresh all data">
              <IconButton onClick={refreshData} disabled={loading}>
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Quick Stats */}
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={6} sm={3}>
            <Card variant="outlined">
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <DataObject color="primary" />
                  <Box>
                    <Typography variant="h6">{data.entities.length}</Typography>
                    <Typography variant="caption" color="text.secondary">Entities</Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card variant="outlined">
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <AccountTree color="secondary" />
                  <Box>
                    <Typography variant="h6">{data.triples.length}</Typography>
                    <Typography variant="caption" color="text.secondary">Triples</Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card variant="outlined">
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Analytics color="success" />
                  <Box>
                    <Typography variant="h6">{data.relationships.length}</Typography>
                    <Typography variant="caption" color="text.secondary">Relationships</Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card variant="outlined">
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Storage color="warning" />
                  <Box>
                    <Typography variant="h6">{data.databases.length}</Typography>
                    <Typography variant="caption" color="text.secondary">Databases</Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      {/* Tabs */}
      <Paper sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label={`Entities (${data.entities.length})`} icon={<DataObject />} />
          <Tab label={`Triples (${data.triples.length})`} icon={<AccountTree />} />
          <Tab label={`Relationships (${data.relationships.length})`} icon={<Analytics />} />
          <Tab label={`Text Chunks (${data.chunks.length})`} icon={<Article />} />
          <Tab label={`Databases (${data.databases.length})`} icon={<Storage />} />
          <Tab label="Entity Details" icon={<Visibility />} disabled={!selectedEntity} />
        </Tabs>

        {/* Tab Panels */}
        <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
          {/* Entities Tab */}
          <TabPanel value={tabValue} index={0}>
            <Typography variant="h6" gutterBottom>
              All Entities in Knowledge Graph
              {data.entity_types.length > 0 && (
                <Box component="span" sx={{ ml: 2 }}>
                  {data.entity_types.map(type => (
                    <Chip key={type} label={type} size="small" sx={{ ml: 0.5 }} />
                  ))}
                </Box>
              )}
            </Typography>
            
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
              <Table stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell>Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Description</TableCell>
                    <TableCell>Properties</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredEntities.map((entity, index) => (
                    <TableRow 
                      key={entity.id} 
                      hover 
                      onClick={() => handleEntitySelect(entity)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {entity.id}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" fontWeight="medium">
                          {entity.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip label={entity.entity_type} size="small" color="primary" />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                          {entity.description || 'No description'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Badge badgeContent={Object.keys(entity.properties).length} color="secondary">
                          <Button size="small" startIcon={<Launch />}>
                            View Properties
                          </Button>
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Button
                          size="small"
                          onClick={(e: React.MouseEvent) => {
                            e.stopPropagation();
                            handleEntitySelect(entity);
                          }}
                        >
                          Details
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Triples Tab */}
          <TabPanel value={tabValue} index={1}>
            <Typography variant="h6" gutterBottom>
              All Knowledge Triples (Subject-Predicate-Object)
            </Typography>
            
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
              <Table stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>Subject</TableCell>
                    <TableCell>Predicate</TableCell>
                    <TableCell>Object</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Source</TableCell>
                    <TableCell>Metadata</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredTriples.map((triple, index) => (
                    <TableRow key={index} hover>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {triple.subject}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip label={triple.predicate} size="small" color="secondary" />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {triple.object}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {triple.confidence.toFixed(3)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {triple.source || 'Unknown'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {triple.metadata && Object.keys(triple.metadata).length > 0 ? (
                          <Button size="small" startIcon={<DataObject />}>
                            View Metadata
                          </Button>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            None
                          </Typography>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Relationships Tab */}
          <TabPanel value={tabValue} index={2}>
            <Typography variant="h6" gutterBottom>
              All Entity Relationships
            </Typography>
            
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
              <Table stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>Source Entity</TableCell>
                    <TableCell>Relationship Type</TableCell>
                    <TableCell>Target Entity</TableCell>
                    <TableCell>Weight/Strength</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Metadata</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.relationships.map((relationship, index) => (
                    <TableRow key={index} hover>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {relationship.source}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip label={relationship.relationship_type} size="small" color="primary" />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {relationship.target}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {relationship.weight.toFixed(3)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {relationship.confidence?.toFixed(3) || 'N/A'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {relationship.metadata && Object.keys(relationship.metadata).length > 0 ? (
                          <Button size="small" startIcon={<DataObject />}>
                            View Metadata
                          </Button>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            None
                          </Typography>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Text Chunks Tab */}
          <TabPanel value={tabValue} index={3}>
            <Typography variant="h6" gutterBottom>
              All Text Chunks
            </Typography>
            
            <Grid container spacing={2}>
              {data.chunks.map((chunk, index) => (
                <Grid item xs={12} md={6} lg={4} key={chunk.id}>
                  <Card variant="outlined">
                    <CardHeader 
                      title={`Chunk ${index + 1}`}
                      subheader={chunk.id}
                      titleTypographyProps={{ variant: 'subtitle2' }}
                      subheaderTypographyProps={{ 
                        variant: 'caption', 
                        sx: { fontFamily: 'monospace' } 
                      }}
                    />
                    <CardContent sx={{ pt: 0 }}>
                      <Typography variant="body2" sx={{ mb: 2 }}>
                        {chunk.text.substring(0, 200)}
                        {chunk.text.length > 200 && '...'}
                      </Typography>
                      {chunk.embedding && (
                        <Chip 
                          label={`Embedding: ${chunk.embedding.length}D`} 
                          size="small" 
                          color="secondary" 
                        />
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </TabPanel>

          {/* Databases Tab */}
          <TabPanel value={tabValue} index={4}>
            <Typography variant="h6" gutterBottom>
              Knowledge Graph Databases
            </Typography>
            
            <Grid container spacing={3}>
              {data.databases.map((database, index) => (
                <Grid item xs={12} md={6} key={index}>
                  <Card>
                    <CardHeader
                      title={database.name}
                      subheader={database.type}
                      avatar={<Storage color="primary" />}
                    />
                    <CardContent>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Entities
                          </Typography>
                          <Typography variant="h6">
                            {database.entity_count.toLocaleString()}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Triples
                          </Typography>
                          <Typography variant="h6">
                            {database.triple_count.toLocaleString()}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Chunks
                          </Typography>
                          <Typography variant="h6">
                            {database.chunk_count.toLocaleString()}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Size
                          </Typography>
                          <Typography variant="h6">
                            {(database.size_bytes / 1024 / 1024).toFixed(1)} MB
                          </Typography>
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="body2" color="text.secondary">
                            Last Updated
                          </Typography>
                          <Typography variant="body2">
                            {new Date(database.last_updated).toLocaleString()}
                          </Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>

            {/* Overall Statistics */}
            <Card sx={{ mt: 3 }}>
              <CardHeader title="Overall Statistics" />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      Content Distribution
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Total Entities"
                          secondary={data.stats.total_entities.toLocaleString()}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Total Triples"
                          secondary={data.stats.total_triples.toLocaleString()}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Total Relationships"
                          secondary={data.stats.total_relationships.toLocaleString()}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Total Chunks"
                          secondary={data.stats.total_chunks.toLocaleString()}
                        />
                      </ListItem>
                    </List>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      Performance Metrics
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Memory Usage"
                          secondary={`${data.stats.memory_usage.toFixed(1)} MB`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Avg Query Time"
                          secondary={`${data.stats.query_performance.avg_query_time_ms.toFixed(2)} ms`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Total Queries"
                          secondary={data.stats.query_performance.total_queries.toLocaleString()}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Cache Hit Rate"
                          secondary={`${(data.stats.query_performance.cache_hit_rate * 100).toFixed(1)}%`}
                        />
                      </ListItem>
                    </List>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </TabPanel>

          {/* Entity Details Tab */}
          <TabPanel value={tabValue} index={5}>
            {selectedEntity ? (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Entity Details: {selectedEntity.name}
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardHeader title="Basic Information" />
                      <CardContent>
                        <List>
                          <ListItem>
                            <ListItemText 
                              primary="ID"
                              secondary={
                                <Typography sx={{ fontFamily: 'monospace' }}>
                                  {selectedEntity.id}
                                </Typography>
                              }
                            />
                          </ListItem>
                          <ListItem>
                            <ListItemText 
                              primary="Name"
                              secondary={selectedEntity.name}
                            />
                          </ListItem>
                          <ListItem>
                            <ListItemText 
                              primary="Type"
                              secondary={
                                <Chip label={selectedEntity.entity_type} size="small" color="primary" />
                              }
                            />
                          </ListItem>
                          <ListItem>
                            <ListItemText 
                              primary="Description"
                              secondary={selectedEntity.description || 'No description available'}
                            />
                          </ListItem>
                        </List>
                      </CardContent>
                    </Card>

                    {/* Properties */}
                    {Object.keys(selectedEntity.properties).length > 0 && (
                      <Card sx={{ mt: 2 }}>
                        <CardHeader title="Properties" />
                        <CardContent>
                          <List dense>
                            {Object.entries(selectedEntity.properties).map(([key, value]) => (
                              <ListItem key={key}>
                                <ListItemText 
                                  primary={key}
                                  secondary={typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                />
                              </ListItem>
                            ))}
                          </List>
                        </CardContent>
                      </Card>
                    )}
                  </Grid>

                  <Grid item xs={12} md={6}>
                    {/* Related Triples */}
                    {entityDetails?.triples && entityDetails.triples.length > 0 && (
                      <Card>
                        <CardHeader title={`Related Triples (${entityDetails.triples.length})`} />
                        <CardContent>
                          <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
                            {entityDetails.triples.map((triple: RealTriple, index: number) => (
                              <ListItem key={index}>
                                <ListItemText 
                                  primary={`${triple.predicate}: ${triple.object}`}
                                  secondary={`Confidence: ${triple.confidence.toFixed(3)}`}
                                />
                              </ListItem>
                            ))}
                          </List>
                        </CardContent>
                      </Card>
                    )}

                    {/* Related Relationships */}
                    {entityDetails?.relationships && entityDetails.relationships.length > 0 && (
                      <Card sx={{ mt: 2 }}>
                        <CardHeader title={`Relationships (${entityDetails.relationships.length})`} />
                        <CardContent>
                          <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
                            {entityDetails.relationships.map((rel: RealRelationship, index: number) => (
                              <ListItem key={index}>
                                <ListItemText 
                                  primary={`${rel.relationship_type} → ${rel.target}`}
                                  secondary={`Weight: ${rel.weight.toFixed(3)}`}
                                />
                              </ListItem>
                            ))}
                          </List>
                        </CardContent>
                      </Card>
                    )}
                  </Grid>
                </Grid>
              </Box>
            ) : (
              <Alert severity="info">
                Select an entity from the Entities tab to view detailed information.
              </Alert>
            )}
          </TabPanel>
        </Box>
      </Paper>
    </Box>
  );
};

export default RealKnowledgeGraphPage;