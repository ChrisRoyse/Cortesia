/**
 * Layout utility functions for Phase 5 visualization
 */

import { Position, Size, ArchitectureNode, LayerDefinition } from '../types';

/**
 * Calculates the optimal position for a node within a layer
 * @param nodeIndex - Index of the node within its layer
 * @param totalNodes - Total number of nodes in the layer
 * @param layerBounds - Bounds of the layer
 * @returns Calculated position
 */
export function calculateNodePosition(
  nodeIndex: number,
  totalNodes: number,
  layerBounds: { position: Position; size: Size }
): Position {
  const { position, size } = layerBounds;
  
  // Calculate spacing
  const padding = 50;
  const availableWidth = size.width - (padding * 2);
  const nodeSpacing = availableWidth / Math.max(totalNodes - 1, 1);
  
  return {
    x: position.x + padding + (nodeIndex * nodeSpacing),
    y: position.y + size.height / 2
  };
}

/**
 * Arranges nodes in a hierarchical layout
 * @param nodes - Array of nodes to arrange
 * @param layers - Layer definitions
 * @returns Updated nodes with positions
 */
export function hierarchicalLayout(
  nodes: ArchitectureNode[],
  layers: LayerDefinition[]
): ArchitectureNode[] {
  const nodesByLayer = new Map<string, ArchitectureNode[]>();
  
  // Group nodes by layer
  nodes.forEach(node => {
    if (!nodesByLayer.has(node.layer)) {
      nodesByLayer.set(node.layer, []);
    }
    nodesByLayer.get(node.layer)!.push(node);
  });
  
  // Calculate positions for each layer
  return nodes.map(node => {
    const layer = layers.find(l => l.id === node.layer);
    if (!layer) return node;
    
    const layerNodes = nodesByLayer.get(node.layer) || [];
    const nodeIndex = layerNodes.indexOf(node);
    
    const position = calculateNodePosition(
      nodeIndex,
      layerNodes.length,
      { position: layer.position, size: layer.size }
    );
    
    return { ...node, position };
  });
}

/**
 * Arranges nodes in a force-directed layout
 * @param nodes - Array of nodes to arrange
 * @param connections - Connection information
 * @param bounds - Layout bounds
 * @returns Updated nodes with positions
 */
export function forceDirectedLayout(
  nodes: ArchitectureNode[],
  connections: Array<{ sourceId: string; targetId: string; strength: number }>,
  bounds: { width: number; height: number }
): ArchitectureNode[] {
  const iterations = 50;
  const repulsionStrength = 1000;
  const attractionStrength = 0.1;
  const damping = 0.9;
  
  // Initialize velocities
  const velocities = new Map<string, Position>();
  nodes.forEach(node => velocities.set(node.id, { x: 0, y: 0 }));
  
  // Create a mutable copy of nodes
  let layoutNodes = nodes.map(node => ({
    ...node,
    position: {
      x: node.position.x || Math.random() * bounds.width,
      y: node.position.y || Math.random() * bounds.height
    }
  }));
  
  // Run simulation
  for (let i = 0; i < iterations; i++) {
    // Apply repulsion between all nodes
    for (let j = 0; j < layoutNodes.length; j++) {
      for (let k = j + 1; k < layoutNodes.length; k++) {
        const nodeA = layoutNodes[j];
        const nodeB = layoutNodes[k];
        
        const dx = nodeB.position.x - nodeA.position.x;
        const dy = nodeB.position.y - nodeA.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
        
        const force = repulsionStrength / (distance * distance);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        
        const velA = velocities.get(nodeA.id)!;
        const velB = velocities.get(nodeB.id)!;
        
        velA.x -= fx;
        velA.y -= fy;
        velB.x += fx;
        velB.y += fy;
      }
    }
    
    // Apply attraction along connections
    connections.forEach(conn => {
      const sourceNode = layoutNodes.find(n => n.id === conn.sourceId);
      const targetNode = layoutNodes.find(n => n.id === conn.targetId);
      
      if (sourceNode && targetNode) {
        const dx = targetNode.position.x - sourceNode.position.x;
        const dy = targetNode.position.y - sourceNode.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
        
        const force = distance * attractionStrength * conn.strength;
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        
        const velSource = velocities.get(sourceNode.id)!;
        const velTarget = velocities.get(targetNode.id)!;
        
        velSource.x += fx;
        velSource.y += fy;
        velTarget.x -= fx;
        velTarget.y -= fy;
      }
    });
    
    // Update positions
    layoutNodes = layoutNodes.map(node => {
      const vel = velocities.get(node.id)!;
      
      // Apply damping
      vel.x *= damping;
      vel.y *= damping;
      
      // Update position
      const newX = Math.max(50, Math.min(bounds.width - 50, node.position.x + vel.x));
      const newY = Math.max(50, Math.min(bounds.height - 50, node.position.y + vel.y));
      
      return {
        ...node,
        position: { x: newX, y: newY }
      };
    });
  }
  
  return layoutNodes;
}

/**
 * Arranges nodes in a circular layout
 * @param nodes - Array of nodes to arrange
 * @param center - Center point of the circle
 * @param radius - Radius of the circle
 * @returns Updated nodes with positions
 */
export function circularLayout(
  nodes: ArchitectureNode[],
  center: Position,
  radius: number
): ArchitectureNode[] {
  const angleStep = (2 * Math.PI) / nodes.length;
  
  return nodes.map((node, index) => {
    const angle = index * angleStep - Math.PI / 2; // Start from top
    
    return {
      ...node,
      position: {
        x: center.x + radius * Math.cos(angle),
        y: center.y + radius * Math.sin(angle)
      }
    };
  });
}

/**
 * Arranges nodes in a grid layout
 * @param nodes - Array of nodes to arrange
 * @param bounds - Layout bounds
 * @param columns - Number of columns (optional, auto-calculated if not provided)
 * @returns Updated nodes with positions
 */
export function gridLayout(
  nodes: ArchitectureNode[],
  bounds: { width: number; height: number },
  columns?: number
): ArchitectureNode[] {
  const nodeCount = nodes.length;
  const cols = columns || Math.ceil(Math.sqrt(nodeCount));
  const rows = Math.ceil(nodeCount / cols);
  
  const cellWidth = bounds.width / cols;
  const cellHeight = bounds.height / rows;
  
  return nodes.map((node, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    
    return {
      ...node,
      position: {
        x: (col + 0.5) * cellWidth,
        y: (row + 0.5) * cellHeight
      }
    };
  });
}

/**
 * Calculates the bounding box of a set of nodes
 * @param nodes - Array of nodes
 * @returns Bounding box
 */
export function calculateBoundingBox(nodes: ArchitectureNode[]): {
  min: Position;
  max: Position;
  width: number;
  height: number;
} {
  if (nodes.length === 0) {
    return {
      min: { x: 0, y: 0 },
      max: { x: 0, y: 0 },
      width: 0,
      height: 0
    };
  }
  
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  
  nodes.forEach(node => {
    minX = Math.min(minX, node.position.x - node.size);
    minY = Math.min(minY, node.position.y - node.size);
    maxX = Math.max(maxX, node.position.x + node.size);
    maxY = Math.max(maxY, node.position.y + node.size);
  });
  
  return {
    min: { x: minX, y: minY },
    max: { x: maxX, y: maxY },
    width: maxX - minX,
    height: maxY - minY
  };
}

/**
 * Centers nodes within given bounds
 * @param nodes - Array of nodes
 * @param bounds - Target bounds
 * @returns Updated nodes with centered positions
 */
export function centerNodes(
  nodes: ArchitectureNode[],
  bounds: { width: number; height: number }
): ArchitectureNode[] {
  const bbox = calculateBoundingBox(nodes);
  
  const offsetX = (bounds.width - bbox.width) / 2 - bbox.min.x;
  const offsetY = (bounds.height - bbox.height) / 2 - bbox.min.y;
  
  return nodes.map(node => ({
    ...node,
    position: {
      x: node.position.x + offsetX,
      y: node.position.y + offsetY
    }
  }));
}