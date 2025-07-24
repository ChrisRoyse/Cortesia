import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, Sphere, Line, Text } from '@react-three/drei';
import { Box, Paper, Typography, Slider, Switch, TextField, Button, Chip, Grid, Card, CardContent, IconButton, Tooltip, FormControlLabel, LinearProgress } from '@mui/material';
import { Search, Download, PlayArrow, Pause, Refresh, FilterList, Timeline, BubbleChart } from '@mui/icons-material';
import { BrainGraphData, BrainEntity, BrainRelationship, ConceptStructure } from '../../types/brain';
import * as THREE from 'three';

interface BrainKnowledgeGraphProps {
  data?: BrainGraphData;
  height?: number | string;
  onEntitySelect?: (entity: BrainEntity) => void;
  onRelationshipSelect?: (relationship: BrainRelationship) => void;
}

// Entity node component
const EntityNode: React.FC<{
  entity: BrainEntity;
  position: [number, number, number];
  onClick: () => void;
  isSelected: boolean;
  isHighlighted: boolean;
}> = ({ entity, position, onClick, isSelected, isHighlighted }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  // Animate activation level
  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.scale.setScalar(1 + entity.activation * 0.5);
      if (isSelected) {
        meshRef.current.rotation.y += delta * 0.5;
      }
    }
  });

  const color = useMemo(() => {
    switch (entity.direction) {
      case 'Input': return '#2196f3';
      case 'Output': return '#4caf50';
      case 'Gate': return '#ff9800';
      case 'Hidden': return '#9c27b0';
      default: return '#607d8b';
    }
  }, [entity.direction]);

  return (
    <group position={position}>
      <Sphere
        ref={meshRef}
        args={[0.5, 32, 32]}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={entity.activation}
          opacity={0.8 + entity.activation * 0.2}
          transparent
          wireframe={hovered}
        />
      </Sphere>
      {(hovered || isSelected) && (
        <Html>
          <div style={{
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            whiteSpace: 'nowrap'
          }}>
            <div>{entity.properties.name || entity.id}</div>
            <div>Activation: {entity.activation.toFixed(2)}</div>
            <div>Type: {entity.direction}</div>
          </div>
        </Html>
      )}
    </group>
  );
};

// Relationship edge component
const RelationshipEdge: React.FC<{
  relationship: BrainRelationship;
  fromPos: [number, number, number];
  toPos: [number, number, number];
  onClick: () => void;
}> = ({ relationship, fromPos, toPos, onClick }) => {
  const points = useMemo(() => [
    new THREE.Vector3(...fromPos),
    new THREE.Vector3(...toPos)
  ], [fromPos, toPos]);

  const color = relationship.inhibitory ? '#f44336' : '#2196f3';
  const lineWidth = 1 + relationship.weight * 4;

  return (
    <Line
      points={points}
      color={color}
      lineWidth={lineWidth}
      onClick={onClick}
      opacity={0.6 + relationship.weight * 0.4}
      transparent
      dashed={relationship.inhibitory}
    />
  );
};

// 3D Graph Scene
const GraphScene: React.FC<{
  data: BrainGraphData;
  selectedEntity: string | null;
  selectedRelationship: string | null;
  onEntityClick: (entity: BrainEntity) => void;
  onRelationshipClick: (relationship: BrainRelationship) => void;
  activationThreshold: number;
  showInhibitory: boolean;
  searchTerm: string;
  animatePropagation: boolean;
}> = ({ 
  data, 
  selectedEntity, 
  selectedRelationship,
  onEntityClick, 
  onRelationshipClick,
  activationThreshold,
  showInhibitory,
  searchTerm,
  animatePropagation
}) => {
  const { camera } = useThree();
  
  // Calculate entity positions using force-directed layout
  const entityPositions = useMemo(() => {
    const positions = new Map<string, [number, number, number]>();
    const radius = 10;
    
    data.entities.forEach((entity, index) => {
      const angle = (index / data.entities.length) * Math.PI * 2;
      const layerOffset = {
        'Input': -radius,
        'Hidden': 0,
        'Gate': radius * 0.5,
        'Output': radius
      }[entity.direction] || 0;
      
      positions.set(entity.id, [
        Math.cos(angle) * radius + (Math.random() - 0.5) * 2,
        layerOffset,
        Math.sin(angle) * radius + (Math.random() - 0.5) * 2
      ]);
    });
    
    return positions;
  }, [data.entities]);

  // Filter entities based on search and activation threshold
  const filteredEntities = useMemo(() => {
    return data.entities.filter(entity => {
      const matchesSearch = !searchTerm || 
        entity.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        JSON.stringify(entity.properties).toLowerCase().includes(searchTerm.toLowerCase());
      const meetsThreshold = entity.activation >= activationThreshold;
      return matchesSearch && meetsThreshold;
    });
  }, [data.entities, searchTerm, activationThreshold]);

  // Filter relationships
  const filteredRelationships = useMemo(() => {
    const entityIds = new Set(filteredEntities.map(e => e.id));
    return data.relationships.filter(rel => {
      const isConnected = entityIds.has(rel.from) && entityIds.has(rel.to);
      const showRel = showInhibitory || !rel.inhibitory;
      return isConnected && showRel;
    });
  }, [data.relationships, filteredEntities, showInhibitory]);

  // Animation state
  const animationRef = useRef(0);
  useFrame((state, delta) => {
    if (animatePropagation) {
      animationRef.current += delta;
      // Implement activation propagation animation
    }
  });

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <OrbitControls enablePan enableZoom enableRotate />
      
      {/* Render entities */}
      {filteredEntities.map((entity) => (
        <EntityNode
          key={entity.id}
          entity={entity}
          position={entityPositions.get(entity.id) || [0, 0, 0]}
          onClick={() => onEntityClick(entity)}
          isSelected={selectedEntity === entity.id}
          isHighlighted={false}
        />
      ))}
      
      {/* Render relationships */}
      {filteredRelationships.map((rel, index) => (
        <RelationshipEdge
          key={`${rel.from}-${rel.to}-${index}`}
          relationship={rel}
          fromPos={entityPositions.get(rel.from) || [0, 0, 0]}
          toPos={entityPositions.get(rel.to) || [0, 0, 0]}
          onClick={() => onRelationshipClick(rel)}
        />
      ))}
      
      {animatePropagation && (
        <mesh visible={false}>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial data-testid="propagation-animation" />
        </mesh>
      )}
    </>
  );
};

export const BrainKnowledgeGraph: React.FC<BrainKnowledgeGraphProps> = ({
  data,
  height = 600,
  onEntitySelect,
  onRelationshipSelect
}) => {
  const [selectedEntity, setSelectedEntity] = useState<BrainEntity | null>(null);
  const [selectedRelationship, setSelectedRelationship] = useState<BrainRelationship | null>(null);
  const [activationThreshold, setActivationThreshold] = useState(0);
  const [showInhibitory, setShowInhibitory] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [animatePropagation, setAnimatePropagation] = useState(false);
  const [showTimeline, setShowTimeline] = useState(false);

  const handleEntityClick = useCallback((entity: BrainEntity) => {
    setSelectedEntity(entity);
    setSelectedRelationship(null);
    onEntitySelect?.(entity);
  }, [onEntitySelect]);

  const handleRelationshipClick = useCallback((relationship: BrainRelationship) => {
    setSelectedRelationship(relationship);
    setSelectedEntity(null);
    onRelationshipSelect?.(relationship);
  }, [onRelationshipSelect]);

  const handleExport = useCallback(() => {
    if (!data) return;
    
    const exportData = {
      entities: data.entities,
      relationships: data.relationships,
      statistics: data.statistics,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `brain-graph-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [data]);

  if (!data) {
    return (
      <Paper sx={{ p: 3, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          No brain graph data available
        </Typography>
      </Paper>
    );
  }

  const selectedConcept = selectedEntity && data.concepts.find(c => 
    c.inputs.includes(selectedEntity.id) || 
    c.outputs.includes(selectedEntity.id) || 
    c.gates.includes(selectedEntity.id)
  );

  return (
    <Box sx={{ height, display: 'flex', flexDirection: 'column' }}>
      <Grid container spacing={2} sx={{ flexGrow: 1 }}>
        {/* 3D Visualization */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ height: '100%', position: 'relative' }}>
            <Canvas camera={{ position: [0, 5, 20], fov: 60 }}>
              <GraphScene
                data={data}
                selectedEntity={selectedEntity?.id || null}
                selectedRelationship={selectedRelationship ? `${selectedRelationship.from}-${selectedRelationship.to}` : null}
                onEntityClick={handleEntityClick}
                onRelationshipClick={handleRelationshipClick}
                activationThreshold={activationThreshold}
                showInhibitory={showInhibitory}
                searchTerm={searchTerm}
                animatePropagation={animatePropagation}
              />
            </Canvas>
            
            {/* Controls Overlay */}
            <Box sx={{ position: 'absolute', top: 16, left: 16, right: 16 }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={4}>
                  <TextField
                    size="small"
                    placeholder="Search entities..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    InputProps={{
                      startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />
                    }}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={12} sm={8}>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={showInhibitory}
                          onChange={(e) => setShowInhibitory(e.target.checked)}
                        />
                      }
                      label="Show Inhibitory"
                    />
                    <Button
                      size="small"
                      startIcon={animatePropagation ? <Pause /> : <PlayArrow />}
                      onClick={() => setAnimatePropagation(!animatePropagation)}
                    >
                      {animatePropagation ? 'Pause' : 'Animate'} Propagation
                    </Button>
                    <Button
                      size="small"
                      startIcon={<Timeline />}
                      onClick={() => setShowTimeline(!showTimeline)}
                    >
                      Activation History
                    </Button>
                    <IconButton size="small" onClick={handleExport}>
                      <Download />
                    </IconButton>
                  </Box>
                </Grid>
              </Grid>
            </Box>
          </Paper>
        </Grid>

        {/* Side Panels */}
        <Grid item xs={12} md={4}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
            {/* Entity Details */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Entity Details</Typography>
                {selectedEntity ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Typography variant="body2">
                      <strong>ID:</strong> {selectedEntity.id}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Type:</strong> <Chip label={selectedEntity.direction} size="small" />
                    </Typography>
                    <Typography variant="body2">
                      <strong>Activation:</strong>
                      <LinearProgress 
                        variant="determinate" 
                        value={selectedEntity.activation * 100} 
                        sx={{ mt: 0.5 }}
                      />
                    </Typography>
                    {selectedConcept && (
                      <>
                        <Typography variant="body2">
                          <strong>Concept:</strong> {selectedConcept.name}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Coherence:</strong> {selectedConcept.coherence.toFixed(2)}
                        </Typography>
                      </>
                    )}
                    <Typography variant="body2">
                      <strong>Properties:</strong>
                    </Typography>
                    <pre style={{ fontSize: '12px', overflow: 'auto' }}>
                      {JSON.stringify(selectedEntity.properties, null, 2)}
                    </pre>
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Select an entity to view details
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
                    <Typography variant="body2">Entities: {data.statistics.entityCount}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">Relationships: {data.statistics.relationshipCount}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">Avg Activation: {data.statistics.avgActivation.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">Graph Density: {data.statistics.graphDensity.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">Clustering: {data.statistics.clusteringCoefficient.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">Learning Eff: {data.statistics.learningEfficiency.toFixed(2)}</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            {/* Filters */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Filters</Typography>
                <Typography variant="body2" gutterBottom>
                  Activation Threshold: {activationThreshold.toFixed(2)}
                </Typography>
                <Slider
                  value={activationThreshold}
                  onChange={(_, value: number | number[]) => setActivationThreshold(value as number)}
                  min={0}
                  max={1}
                  step={0.1}
                  marks
                  valueLabelDisplay="auto"
                  aria-label="activation threshold"
                />
              </CardContent>
            </Card>
          </Box>
        </Grid>
      </Grid>

      {/* Timeline Modal */}
      {showTimeline && (
        <Paper 
          sx={{ 
            position: 'absolute', 
            top: '50%', 
            left: '50%', 
            transform: 'translate(-50%, -50%)',
            p: 3,
            zIndex: 1000
          }}
        >
          <Typography variant="h6">Activation Timeline</Typography>
          {/* Timeline implementation */}
        </Paper>
      )}
    </Box>
  );
};