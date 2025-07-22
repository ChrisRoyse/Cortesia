import React, { useRef, useEffect, useMemo, useCallback, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import * as THREE from 'three';
import { useAppSelector } from '../../stores';
import { KnowledgeNode, KnowledgeEdge } from '../../types';

// Extended types for 3D positioning
export interface GraphNode3D extends KnowledgeNode {
  position3D: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
  force: { x: number; y: number; z: number };
}

export interface GraphEdge3D extends KnowledgeEdge {
  sourceNode?: GraphNode3D;
  targetNode?: GraphNode3D;
}

interface KnowledgeGraph3DProps {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  onNodeClick?: (node: KnowledgeNode) => void;
  onNodeHover?: (node: KnowledgeNode | null) => void;
  interactive?: boolean;
  width?: number;
  height?: number;
  enablePhysics?: boolean;
  showLabels?: boolean;
  nodeScale?: number;
  className?: string;
}

// Force simulation parameters
const FORCE_CONFIG = {
  centerStrength: 0.1,
  linkStrength: 0.3,
  linkDistance: 50,
  chargeStrength: -300,
  velocityDecay: 0.4,
  alphaDecay: 0.02,
  alphaMin: 0.001,
};

// Node component with physics simulation
const Node3D: React.FC<{
  node: GraphNode3D;
  scale: number;
  isHovered: boolean;
  isSelected: boolean;
  onPointerOver: () => void;
  onPointerOut: () => void;
  onClick: () => void;
  showLabel: boolean;
}> = ({ node, scale, isHovered, isSelected, onPointerOver, onPointerOut, onClick, showLabel }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const theme = useAppSelector(state => state.dashboard.config.theme);
  
  // Node color based on type and state
  const nodeColor = useMemo(() => {
    if (isSelected) return '#ff6b35';
    if (isHovered) return '#4ecdc4';
    
    switch (node.type) {
      case 'concept': return theme === 'dark' ? '#61dafb' : '#0070f3';
      case 'entity': return theme === 'dark' ? '#f093fb' : '#8b5cf6';
      case 'relation': return theme === 'dark' ? '#a8e6cf' : '#10b981';
      case 'property': return theme === 'dark' ? '#ffd93d' : '#f59e0b';
      default: return theme === 'dark' ? '#ffffff' : '#374151';
    }
  }, [node.type, isHovered, isSelected, theme]);

  // Node size based on weight
  const nodeSize = useMemo(() => {
    const baseSize = scale * 0.5;
    const weightMultiplier = Math.sqrt(node.weight || 1);
    return Math.max(baseSize * weightMultiplier, baseSize * 0.3);
  }, [node.weight, scale]);

  // Update position based on physics simulation
  useFrame((_, delta) => {
    if (meshRef.current && node.position3D) {
      meshRef.current.position.set(
        node.position3D.x,
        node.position3D.y,
        node.position3D.z
      );
    }
  });

  return (
    <group>
      <mesh
        ref={meshRef}
        position={[node.position3D.x, node.position3D.y, node.position3D.z]}
        onPointerOver={onPointerOver}
        onPointerOut={onPointerOut}
        onClick={onClick}
      >
        <sphereGeometry args={[nodeSize, 16, 16]} />
        <meshStandardMaterial 
          color={nodeColor}
          transparent
          opacity={isHovered ? 0.9 : 0.7}
          roughness={0.3}
          metalness={0.1}
        />
      </mesh>
      
      {showLabel && (isHovered || isSelected) && (
        <Html
          position={[node.position3D.x, node.position3D.y + nodeSize + 5, node.position3D.z]}
          center
          distanceFactor={10}
        >
          <div
            style={{
              background: theme === 'dark' ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.9)',
              color: theme === 'dark' ? '#ffffff' : '#374151',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '12px',
              whiteSpace: 'nowrap',
              border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
            }}
          >
            {node.label}
          </div>
        </Html>
      )}
    </group>
  );
};

// Edge component
const Edge3D: React.FC<{
  edge: GraphEdge3D;
  opacity: number;
}> = ({ edge, opacity }) => {
  const lineRef = useRef<THREE.BufferGeometry>(null);
  const theme = useAppSelector(state => state.dashboard.config.theme);

  const edgeColor = useMemo(() => {
    return theme === 'dark' ? '#6b7280' : '#9ca3af';
  }, [theme]);

  const lineWidth = useMemo(() => {
    return Math.max(edge.weight * 2, 0.5);
  }, [edge.weight]);

  useEffect(() => {
    if (lineRef.current && edge.sourceNode && edge.targetNode) {
      const positions = new Float32Array([
        edge.sourceNode.position3D.x, edge.sourceNode.position3D.y, edge.sourceNode.position3D.z,
        edge.targetNode.position3D.x, edge.targetNode.position3D.y, edge.targetNode.position3D.z,
      ]);
      lineRef.current.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    }
  }, [edge.sourceNode?.position3D, edge.targetNode?.position3D]);

  if (!edge.sourceNode || !edge.targetNode) return null;

  return (
    <line>
      <bufferGeometry ref={lineRef} />
      <lineBasicMaterial 
        color={edgeColor} 
        transparent 
        opacity={opacity * edge.confidence} 
        linewidth={lineWidth}
      />
    </line>
  );
};

// Main 3D scene component
const Graph3DScene: React.FC<{
  nodes: GraphNode3D[];
  edges: GraphEdge3D[];
  onNodeClick?: (node: KnowledgeNode) => void;
  onNodeHover?: (node: KnowledgeNode | null) => void;
  enablePhysics: boolean;
  showLabels: boolean;
  nodeScale: number;
}> = ({ nodes, edges, onNodeClick, onNodeHover, enablePhysics, showLabels, nodeScale }) => {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const { camera } = useThree();
  const animationRef = useRef<number>(0);

  // Physics simulation using useFrame
  useFrame((_, delta) => {
    if (!enablePhysics) return;

    animationRef.current += delta;

    // Apply forces
    nodes.forEach(node => {
      // Reset forces
      node.force = { x: 0, y: 0, z: 0 };

      // Center force
      const centerForce = FORCE_CONFIG.centerStrength;
      node.force.x -= node.position3D.x * centerForce;
      node.force.y -= node.position3D.y * centerForce;
      node.force.z -= node.position3D.z * centerForce;

      // Repulsion force from other nodes
      nodes.forEach(otherNode => {
        if (node.id !== otherNode.id) {
          const dx = node.position3D.x - otherNode.position3D.x;
          const dy = node.position3D.y - otherNode.position3D.y;
          const dz = node.position3D.z - otherNode.position3D.z;
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
          
          const force = FORCE_CONFIG.chargeStrength / (distance * distance);
          node.force.x += (dx / distance) * force;
          node.force.y += (dy / distance) * force;
          node.force.z += (dz / distance) * force;
        }
      });
    });

    // Apply edge forces
    edges.forEach(edge => {
      if (edge.sourceNode && edge.targetNode) {
        const dx = edge.targetNode.position3D.x - edge.sourceNode.position3D.x;
        const dy = edge.targetNode.position3D.y - edge.sourceNode.position3D.y;
        const dz = edge.targetNode.position3D.z - edge.sourceNode.position3D.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
        
        const force = (distance - FORCE_CONFIG.linkDistance) * FORCE_CONFIG.linkStrength;
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        const fz = (dz / distance) * force;
        
        edge.sourceNode.force.x += fx;
        edge.sourceNode.force.y += fy;
        edge.sourceNode.force.z += fz;
        
        edge.targetNode.force.x -= fx;
        edge.targetNode.force.y -= fy;
        edge.targetNode.force.z -= fz;
      }
    });

    // Update velocities and positions
    nodes.forEach(node => {
      node.velocity.x = (node.velocity.x + node.force.x * delta) * FORCE_CONFIG.velocityDecay;
      node.velocity.y = (node.velocity.y + node.force.y * delta) * FORCE_CONFIG.velocityDecay;
      node.velocity.z = (node.velocity.z + node.force.z * delta) * FORCE_CONFIG.velocityDecay;
      
      node.position3D.x += node.velocity.x * delta;
      node.position3D.y += node.velocity.y * delta;
      node.position3D.z += node.velocity.z * delta;
    });
  });

  const handleNodePointerOver = useCallback((node: KnowledgeNode) => {
    setHoveredNode(node.id);
    onNodeHover?.(node);
  }, [onNodeHover]);

  const handleNodePointerOut = useCallback(() => {
    setHoveredNode(null);
    onNodeHover?.(null);
  }, [onNodeHover]);

  const handleNodeClick = useCallback((node: KnowledgeNode) => {
    setSelectedNode(node.id);
    onNodeClick?.(node);
  }, [onNodeClick]);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[100, 100, 100]} intensity={0.8} />
      <pointLight position={[-100, -100, -100]} intensity={0.3} />

      {/* Edges */}
      {edges.map(edge => (
        <Edge3D
          key={edge.id}
          edge={edge}
          opacity={0.6}
        />
      ))}

      {/* Nodes */}
      {nodes.map(node => (
        <Node3D
          key={node.id}
          node={node}
          scale={nodeScale}
          isHovered={hoveredNode === node.id}
          isSelected={selectedNode === node.id}
          onPointerOver={() => handleNodePointerOver(node)}
          onPointerOut={handleNodePointerOut}
          onClick={() => handleNodeClick(node)}
          showLabel={showLabels}
        />
      ))}

      {/* Controls */}
      <OrbitControls
        enableDamping
        dampingFactor={0.1}
        minDistance={10}
        maxDistance={1000}
        enablePan
        enableZoom
        enableRotate
      />
    </>
  );
};

// Main component
export const KnowledgeGraph3D: React.FC<KnowledgeGraph3DProps> = ({
  nodes,
  edges,
  onNodeClick,
  onNodeHover,
  interactive = true,
  width = 800,
  height = 600,
  enablePhysics = true,
  showLabels = true,
  nodeScale = 1,
  className = '',
}) => {
  const theme = useAppSelector(state => state.dashboard.config.theme);

  // Convert nodes to 3D nodes with physics properties
  const nodes3D = useMemo((): GraphNode3D[] => {
    return nodes.map(node => ({
      ...node,
      position3D: {
        x: (node.position.x - 0.5) * 200 + (Math.random() - 0.5) * 20,
        y: (node.position.y - 0.5) * 200 + (Math.random() - 0.5) * 20,
        z: (Math.random() - 0.5) * 100,
      },
      velocity: { x: 0, y: 0, z: 0 },
      force: { x: 0, y: 0, z: 0 },
    }));
  }, [nodes]);

  // Convert edges to 3D edges with node references
  const edges3D = useMemo((): GraphEdge3D[] => {
    return edges.map(edge => ({
      ...edge,
      sourceNode: nodes3D.find(n => n.id === edge.source),
      targetNode: nodes3D.find(n => n.id === edge.target),
    }));
  }, [edges, nodes3D]);

  const canvasStyle = useMemo(() => ({
    width: width,
    height: height,
    background: theme === 'dark' 
      ? 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)' 
      : 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
    borderRadius: '8px',
    border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
  }), [width, height, theme]);

  if (!nodes.length) {
    return (
      <div 
        className={`flex items-center justify-center ${className}`}
        style={canvasStyle}
      >
        <div 
          style={{ 
            color: theme === 'dark' ? '#9ca3af' : '#6b7280',
            fontSize: '14px' 
          }}
        >
          No knowledge graph data available
        </div>
      </div>
    );
  }

  return (
    <div className={className} style={{ width, height }}>
      <Canvas
        style={canvasStyle}
        camera={{ 
          position: [0, 0, 200], 
          fov: 60,
          near: 0.1,
          far: 2000 
        }}
        onCreated={({ gl }) => {
          gl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        }}
      >
        <Graph3DScene
          nodes={nodes3D}
          edges={edges3D}
          onNodeClick={interactive ? onNodeClick : undefined}
          onNodeHover={interactive ? onNodeHover : undefined}
          enablePhysics={enablePhysics}
          showLabels={showLabels}
          nodeScale={nodeScale}
        />
      </Canvas>
    </div>
  );
};

export default KnowledgeGraph3D;