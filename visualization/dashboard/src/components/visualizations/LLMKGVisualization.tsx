import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, Sphere, Line, Text } from '@react-three/drei';
import { 
  Box, 
  Paper, 
  Typography, 
  Slider, 
  Switch, 
  TextField, 
  Button, 
  Chip, 
  Grid, 
  Card, 
  CardContent, 
  IconButton, 
  Tooltip, 
  FormControlLabel, 
  LinearProgress,
  Alert
} from '@mui/material';
import { 
  Search, 
  Download, 
  PlayArrow, 
  Pause, 
  Refresh, 
  FilterList, 
  Timeline, 
  BubbleChart 
} from '@mui/icons-material';
import * as THREE from 'three';
import { 
  ComprehensiveKnowledgeGraphData, 
  RealEntity, 
  RealRelationship,
  RealTriple 
} from '../../services/KnowledgeGraphDataService';

interface LLMKGVisualizationProps {
  data?: ComprehensiveKnowledgeGraphData;
  height?: number | string;
  onEntitySelect?: (entity: RealEntity) => void;
  onRelationshipSelect?: (relationship: RealRelationship) => void;
}

// Enhanced entity node for LLMKG data
const EntityNode: React.FC<{
  entity: RealEntity;
  position: [number, number, number];
  onClick: () => void;
  isSelected: boolean;
  isHighlighted: boolean;
  size: number;
}> = ({ entity, position, onClick, isSelected, isHighlighted, size }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  // Animate selected entities
  useFrame((state, delta) => {
    if (meshRef.current && isSelected) {
      meshRef.current.rotation.y += delta * 0.5;
    }
  });

  // Color based on entity type
  const color = useMemo(() => {
    const typeColors: Record<string, string> = {
      'person': '#e91e63',
      'organization': '#3f51b5',
      'concept': '#4caf50',
      'location': '#ff9800',
      'technology': '#9c27b0',
      'event': '#f44336',
      'document': '#795548',
      'default': '#607d8b'
    };
    return typeColors[entity.entity_type.toLowerCase()] || typeColors.default;
  }, [entity.entity_type]);

  return (
    <group position={position}>
      <Sphere
        ref={meshRef}
        args={[size, 16, 16]}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial
          color={color}
          emissive={isSelected ? color : '#000000'}
          emissiveIntensity={isSelected ? 0.3 : 0}
          opacity={hovered ? 0.9 : 0.7}
          transparent
          wireframe={hovered && !isSelected}
        />
      </Sphere>
      
      {(hovered || isSelected) && (
        <Html>
          <div style={{
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            maxWidth: '200px',
            whiteSpace: 'normal'
          }}>
            <div style={{ fontWeight: 'bold' }}>{entity.name}</div>
            <div style={{ color: '#ccc' }}>Type: {entity.entity_type}</div>
            {entity.description && (
              <div style={{ fontSize: '10px', marginTop: '4px' }}>
                {entity.description.substring(0, 100)}
                {entity.description.length > 100 && '...'}
              </div>
            )}
            <div style={{ fontSize: '10px', color: '#999', marginTop: '4px' }}>
              Properties: {Object.keys(entity.properties).length}
            </div>
          </div>
        </Html>
      )}
    </group>
  );
};

// Relationship edge component for LLMKG data
const RelationshipEdge: React.FC<{
  relationship: RealRelationship;
  fromPos: [number, number, number];
  toPos: [number, number, number];
  onClick: () => void;
  isHighlighted: boolean;
}> = ({ relationship, fromPos, toPos, onClick, isHighlighted }) => {
  const points = useMemo(() => [
    new THREE.Vector3(...fromPos),
    new THREE.Vector3(...toPos)
  ], [fromPos, toPos]);

  const color = isHighlighted ? '#ff6b35' : '#4ecdc4';
  const lineWidth = Math.max(relationship.weight * 3, 0.5);

  return (
    <Line
      points={points}
      color={color}
      lineWidth={lineWidth}
      onClick={onClick}
      opacity={isHighlighted ? 0.9 : 0.6}
      transparent
    />
  );
};

// Main 3D scene component
const LLMKGScene: React.FC<{
  data: ComprehensiveKnowledgeGraphData;
  selectedEntity: RealEntity | null;
  selectedRelationship: RealRelationship | null;
  onEntityClick: (entity: RealEntity) => void;
  onRelationshipClick: (relationship: RealRelationship) => void;
  searchTerm: string;
  minNodeSize: number;
  maxNodeSize: number;
}> = ({ 
  data, 
  selectedEntity, 
  selectedRelationship,
  onEntityClick, 
  onRelationshipClick,
  searchTerm,
  minNodeSize,
  maxNodeSize
}) => {
  const { camera } = useThree();
  
  // Filter entities based on search
  const filteredEntities = useMemo(() => {
    if (!searchTerm) return data.entities;
    const term = searchTerm.toLowerCase();
    return data.entities.filter(entity =>
      entity.name.toLowerCase().includes(term) ||
      entity.entity_type.toLowerCase().includes(term) ||
      entity.description.toLowerCase().includes(term)
    );
  }, [data.entities, searchTerm]);

  // Calculate entity positions using force-directed layout
  const entityPositions = useMemo(() => {
    const positions = new Map<string, [number, number, number]>();
    const radius = Math.max(10, Math.sqrt(filteredEntities.length) * 3);
    
    // Group entities by type for better positioning
    const typeGroups = filteredEntities.reduce((groups, entity) => {
      const type = entity.entity_type;
      if (!groups[type]) groups[type] = [];
      groups[type].push(entity);
      return groups;
    }, {} as Record<string, RealEntity[]>);

    const types = Object.keys(typeGroups);
    
    types.forEach((type, typeIndex) => {
      const typeAngle = (typeIndex / types.length) * Math.PI * 2;
      const typeRadius = radius * 0.7;
      const typeCenter = [
        Math.cos(typeAngle) * typeRadius,
        0,
        Math.sin(typeAngle) * typeRadius
      ];
      
      typeGroups[type].forEach((entity, entityIndex) => {
        const entityAngle = (entityIndex / typeGroups[type].length) * Math.PI * 2;
        const entityRadius = Math.min(radius * 0.3, typeGroups[type].length * 2);
        
        positions.set(entity.id, [
          typeCenter[0] + Math.cos(entityAngle) * entityRadius + (Math.random() - 0.5) * 2,
          typeCenter[1] + (Math.random() - 0.5) * 10,
          typeCenter[2] + Math.sin(entityAngle) * entityRadius + (Math.random() - 0.5) * 2
        ]);
      });
    });
    
    return positions;
  }, [filteredEntities]);

  // Calculate node sizes based on connections
  const entitySizes = useMemo(() => {
    const connectionCounts = new Map<string, number>();
    
    // Count connections for each entity
    data.relationships.forEach(rel => {
      connectionCounts.set(rel.source, (connectionCounts.get(rel.source) || 0) + 1);
      connectionCounts.set(rel.target, (connectionCounts.get(rel.target) || 0) + 1);
    });

    const maxConnections = Math.max(...Array.from(connectionCounts.values()), 1);
    
    return new Map(
      filteredEntities.map(entity => [
        entity.id,
        minNodeSize + ((connectionCounts.get(entity.id) || 0) / maxConnections) * (maxNodeSize - minNodeSize)
      ])
    );
  }, [filteredEntities, data.relationships, minNodeSize, maxNodeSize]);

  // Filter relationships to only show those between visible entities
  const filteredRelationships = useMemo(() => {
    const entityIds = new Set(filteredEntities.map(e => e.id));
    return data.relationships.filter(rel => 
      entityIds.has(rel.source) && entityIds.has(rel.target)
    );
  }, [data.relationships, filteredEntities]);

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[20, 20, 20]} intensity={0.8} />
      <pointLight position={[-20, -20, -20]} intensity={0.3} />
      <OrbitControls enablePan enableZoom enableRotate />
      
      {/* Render entities */}
      {filteredEntities.map((entity) => (
        <EntityNode
          key={entity.id}
          entity={entity}
          position={entityPositions.get(entity.id) || [0, 0, 0]}
          onClick={() => onEntityClick(entity)}
          isSelected={selectedEntity?.id === entity.id}
          isHighlighted={false}
          size={entitySizes.get(entity.id) || minNodeSize}
        />
      ))}
      
      {/* Render relationships */}
      {filteredRelationships.map((rel, index) => (
        <RelationshipEdge
          key={`${rel.source}-${rel.target}-${index}`}
          relationship={rel}
          fromPos={entityPositions.get(rel.source) || [0, 0, 0]}
          toPos={entityPositions.get(rel.target) || [0, 0, 0]}
          onClick={() => onRelationshipClick(rel)}
          isHighlighted={selectedRelationship === rel}
        />
      ))}
    </>
  );
};

export const LLMKGVisualization: React.FC<LLMKGVisualizationProps> = ({
  data,
  height = 600,
  onEntitySelect,
  onRelationshipSelect
}) => {
  const [selectedEntity, setSelectedEntity] = useState<RealEntity | null>(null);
  const [selectedRelationship, setSelectedRelationship] = useState<RealRelationship | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [minNodeSize, setMinNodeSize] = useState(0.5);
  const [maxNodeSize, setMaxNodeSize] = useState(2.0);

  const handleEntityClick = useCallback((entity: RealEntity) => {
    setSelectedEntity(entity);
    setSelectedRelationship(null);
    onEntitySelect?.(entity);
  }, [onEntitySelect]);

  const handleRelationshipClick = useCallback((relationship: RealRelationship) => {
    setSelectedRelationship(relationship);
    setSelectedEntity(null);
    onRelationshipSelect?.(relationship);
  }, [onRelationshipSelect]);

  const handleExport = useCallback(() => {
    if (!data) return;
    
    const exportData = {
      entities: data.entities,
      relationships: data.relationships,
      triples: data.triples,
      stats: data.stats,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `llmkg-graph-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [data]);

  if (!data) {
    return (
      <Paper sx={{ p: 3, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Alert severity="info">
          <Typography variant="h6">No LLMKG data available</Typography>
          <Typography variant="body2">
            Make sure the LLMKG backend is running and contains data.
          </Typography>
        </Alert>
      </Paper>
    );
  }

  if (data.entities.length === 0) {
    return (
      <Paper sx={{ p: 3, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Alert severity="warning">
          <Typography variant="h6">No entities found</Typography>
          <Typography variant="body2">
            The knowledge graph appears to be empty. Add some data using the MCP tools.
          </Typography>
        </Alert>
      </Paper>
    );
  }

  return (
    <Box sx={{ height, display: 'flex', flexDirection: 'column' }}>
      <Grid container spacing={2} sx={{ flexGrow: 1 }}>
        {/* 3D Visualization */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ height: '100%', position: 'relative' }}>
            <Canvas camera={{ position: [0, 10, 30], fov: 60 }}>
              <LLMKGScene
                data={data}
                selectedEntity={selectedEntity}
                selectedRelationship={selectedRelationship}
                onEntityClick={handleEntityClick}
                onRelationshipClick={handleRelationshipClick}
                searchTerm={searchTerm}
                minNodeSize={minNodeSize}
                maxNodeSize={maxNodeSize}
              />
            </Canvas>
            
            {/* Controls Overlay */}
            <Box sx={{ position: 'absolute', top: 16, left: 16, right: 16 }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={6}>
                  <TextField
                    size="small"
                    placeholder="Search entities..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    InputProps={{
                      startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />
                    }}
                    fullWidth
                    sx={{ 
                      backgroundColor: 'rgba(255, 255, 255, 0.9)',
                      borderRadius: 1
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <IconButton 
                      size="small" 
                      onClick={handleExport}
                      sx={{ 
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        '&:hover': {
                          backgroundColor: 'rgba(255, 255, 255, 1)'
                        }
                      }}
                    >
                      <Download />
                    </IconButton>
                  </Box>
                </Grid>
              </Grid>
            </Box>
          </Paper>
        </Grid>

        {/* Side Panel */}
        <Grid item xs={12} md={4}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
            {/* Entity Details */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Entity Details</Typography>
                {selectedEntity ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Typography variant="body2">
                      <strong>Name:</strong> {selectedEntity.name}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Type:</strong> <Chip label={selectedEntity.entity_type} size="small" />
                    </Typography>
                    <Typography variant="body2">
                      <strong>ID:</strong> {selectedEntity.id}
                    </Typography>
                    {selectedEntity.description && (
                      <Typography variant="body2">
                        <strong>Description:</strong> {selectedEntity.description}
                      </Typography>
                    )}
                    <Typography variant="body2">
                      <strong>Properties:</strong> {Object.keys(selectedEntity.properties).length}
                    </Typography>
                    {Object.keys(selectedEntity.properties).length > 0 && (
                      <Box sx={{ maxHeight: 200, overflow: 'auto', fontSize: '12px' }}>
                        <pre style={{ whiteSpace: 'pre-wrap' }}>
                          {JSON.stringify(selectedEntity.properties, null, 2)}
                        </pre>
                      </Box>
                    )}
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Click on an entity to view details
                  </Typography>
                )}
              </CardContent>
            </Card>

            {/* Statistics */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Graph Statistics</Typography>
                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Typography variant="body2">Entities: {data.entities.length}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">Triples: {data.triples.length}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">Relationships: {data.relationships.length}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">Chunks: {data.chunks.length}</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            {/* Controls */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Visualization Controls</Typography>
                <Typography variant="body2" gutterBottom>
                  Node Size Range
                </Typography>
                <Typography variant="caption" gutterBottom>
                  Min: {minNodeSize.toFixed(1)}
                </Typography>
                <Slider
                  value={minNodeSize}
                  onChange={(_, value: number | number[]) => setMinNodeSize(value as number)}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  size="small"
                  sx={{ mb: 1 }}
                />
                <Typography variant="caption" gutterBottom>
                  Max: {maxNodeSize.toFixed(1)}
                </Typography>
                <Slider
                  value={maxNodeSize}
                  onChange={(_, value: number | number[]) => setMaxNodeSize(value as number)}
                  min={1.0}
                  max={5.0}
                  step={0.1}
                  size="small"
                />
              </CardContent>
            </Card>

            {/* Entity Types */}
            {data.entity_types.length > 0 && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Entity Types</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {data.entity_types.map(type => (
                      <Chip 
                        key={type} 
                        label={type} 
                        size="small" 
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </CardContent>
              </Card>
            )}
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default LLMKGVisualization;