import React, { useState, useMemo } from 'react';
import { Box, Typography, Grid, Paper, Alert, CircularProgress, Button } from '@mui/material';
import { Refresh } from '@mui/icons-material';
import { LLMKGVisualization } from '../components/visualizations/LLMKGVisualization';
import { useRealKnowledgeGraph } from '../hooks/useRealKnowledgeGraph';
import { RealEntity, RealRelationship } from '../services/KnowledgeGraphDataService';

const KnowledgeGraphPage: React.FC = () => {
  const [selectedEntity, setSelectedEntity] = useState<RealEntity | null>(null);
  const [selectedRelationship, setSelectedRelationship] = useState<RealRelationship | null>(null);
  
  // Get real LLMKG data with auto-refresh and WebSocket
  const {
    data,
    loading,
    error,
    refreshData,
    connectionStatus,
    lastUpdated
  } = useRealKnowledgeGraph({
    autoRefresh: true,
    refreshInterval: 30000,
    enableWebSocket: true
  });

  const handleEntitySelect = (entity: RealEntity) => {
    setSelectedEntity(entity);
    setSelectedRelationship(null);
  };

  const handleRelationshipSelect = (relationship: RealRelationship) => {
    setSelectedRelationship(relationship);
    setSelectedEntity(null);
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', p: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            LLMKG Knowledge Graph Visualization
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Interactive 3D visualization of entities, relationships, and semantic connections.
            {lastUpdated && (
              <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 2 }}>
                Last updated: {new Date(lastUpdated).toLocaleTimeString()}
              </Typography>
            )}
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={refreshData}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {/* Loading State */}
      {loading && (
        <Paper sx={{ p: 3, mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={24} />
            <Typography>Loading LLMKG data...</Typography>
          </Box>
        </Paper>
      )}

      {/* Error State */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} action={
          <Button onClick={refreshData} startIcon={<Refresh />}>
            Retry
          </Button>
        }>
          <strong>Error loading knowledge graph:</strong> {error}
        </Alert>
      )}

      {/* Connection Status */}
      {connectionStatus !== 'connected' && !loading && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Connection status: {connectionStatus}. Some features may not work properly.
        </Alert>
      )}

      {/* Main Visualization */}
      {data && (
        <Paper sx={{ flexGrow: 1, minHeight: 0 }}>
          <LLMKGVisualization
            data={data}
            height="100%"
            onEntitySelect={handleEntitySelect}
            onRelationshipSelect={handleRelationshipSelect}
          />
        </Paper>
      )}

      {/* Summary Stats when no data */}
      {!data && !loading && !error && (
        <Paper sx={{ p: 3, mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            No Knowledge Graph Data Available
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Make sure the LLMKG backend is running and contains data. You can add data using the MCP tools.
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default KnowledgeGraphPage;