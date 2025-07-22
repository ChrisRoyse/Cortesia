import { ComponentNode, ComponentEdge } from './ArchitectureDiagramEngine';

export interface BrainRegion {
  id: string;
  name: string;
  center: { x: number; y: number; z?: number };
  radius: number;
  phase: number;
  function: 'perception' | 'processing' | 'memory' | 'control' | 'output';
  capacity: number;
  connections: string[];
}

export interface LayoutConstraints {
  minDistance: number;
  maxDistance: number;
  preferredDistance: number;
  hierarchyWeight: number;
  functionWeight: number;
  performanceWeight: number;
}

export interface LayoutMetrics {
  crossings: number;
  totalEdgeLength: number;
  nodeOverlap: number;
  hierarchyViolations: number;
  clusteringCoefficient: number;
  averagePathLength: number;
}

export class LayoutEngine {
  private brainRegions: Map<string, BrainRegion> = new Map();
  private layoutConstraints: LayoutConstraints;
  private dimensions: '2d' | '3d';
  private width: number;
  private height: number;
  private depth: number;

  constructor(
    dimensions: '2d' | '3d' = '2d',
    width = 1200,
    height = 800,
    depth = 600
  ) {
    this.dimensions = dimensions;
    this.width = width;
    this.height = height;
    this.depth = depth;

    this.layoutConstraints = {
      minDistance: 50,
      maxDistance: 300,
      preferredDistance: 120,
      hierarchyWeight: 0.4,
      functionWeight: 0.3,
      performanceWeight: 0.3
    };

    this.initializeBrainRegions();
  }

  private initializeBrainRegions(): void {
    // Define brain-inspired regions for different phases and functions
    const regions: BrainRegion[] = [
      // Phase 1: Perception (Visual Cortex inspired)
      {
        id: 'phase1_perception',
        name: 'Perceptual Input',
        center: { x: this.width * 0.2, y: this.height * 0.3, z: this.depth * 0.2 },
        radius: 100,
        phase: 1,
        function: 'perception',
        capacity: 10,
        connections: ['phase2_processing']
      },
      
      // Phase 2: Processing (Temporal Lobe inspired)
      {
        id: 'phase2_processing',
        name: 'Cognitive Processing',
        center: { x: this.width * 0.4, y: this.height * 0.4, z: this.depth * 0.4 },
        radius: 120,
        phase: 2,
        function: 'processing',
        capacity: 15,
        connections: ['phase1_perception', 'phase3_memory', 'phase4_control']
      },
      
      // Phase 3: Memory (Hippocampus inspired)
      {
        id: 'phase3_memory',
        name: 'Knowledge Storage',
        center: { x: this.width * 0.6, y: this.height * 0.2, z: this.depth * 0.6 },
        radius: 90,
        phase: 3,
        function: 'memory',
        capacity: 20,
        connections: ['phase2_processing', 'phase4_control']
      },
      
      // Phase 4: Control (Prefrontal Cortex inspired)
      {
        id: 'phase4_control',
        name: 'Executive Control',
        center: { x: this.width * 0.7, y: this.height * 0.6, z: this.depth * 0.3 },
        radius: 110,
        phase: 4,
        function: 'control',
        capacity: 12,
        connections: ['phase2_processing', 'phase3_memory', 'phase5_output']
      },
      
      // Phase 5: Output (Motor Cortex inspired)
      {
        id: 'phase5_output',
        name: 'System Output',
        center: { x: this.width * 0.8, y: this.height * 0.7, z: this.depth * 0.8 },
        radius: 80,
        phase: 5,
        function: 'output',
        capacity: 8,
        connections: ['phase4_control']
      }
    ];

    regions.forEach(region => {
      this.brainRegions.set(region.id, region);
    });
  }

  public computeBrainInspiredLayout(
    nodes: ComponentNode[],
    edges: ComponentEdge[]
  ): ComponentNode[] {
    const positionedNodes = nodes.map(node => ({ ...node }));
    
    // Step 1: Assign nodes to brain regions based on phase and function
    const regionAssignments = this.assignNodesToRegions(positionedNodes);
    
    // Step 2: Position nodes within their assigned regions
    this.positionNodesInRegions(positionedNodes, regionAssignments);
    
    // Step 3: Apply force-directed refinement
    this.applyForceDirectedRefinement(positionedNodes, edges);
    
    // Step 4: Optimize for minimal crossings and overlap
    this.optimizeLayout(positionedNodes, edges);
    
    return positionedNodes;
  }

  private assignNodesToRegions(nodes: ComponentNode[]): Map<string, ComponentNode[]> {
    const assignments = new Map<string, ComponentNode[]>();
    
    // Initialize region assignments
    this.brainRegions.forEach((region, regionId) => {
      assignments.set(regionId, []);
    });
    
    nodes.forEach(node => {
      const bestRegion = this.findBestRegionForNode(node);
      const regionNodes = assignments.get(bestRegion) || [];
      regionNodes.push(node);
      assignments.set(bestRegion, regionNodes);
    });
    
    // Balance region loads
    this.balanceRegionLoads(assignments);
    
    return assignments;
  }

  private findBestRegionForNode(node: ComponentNode): string {
    let bestRegion = 'phase2_processing'; // Default
    let bestScore = 0;
    
    this.brainRegions.forEach((region, regionId) => {
      let score = 0;
      
      // Phase matching score
      if (node.phase === region.phase) {
        score += 1.0;
      } else {
        score += Math.max(0, 1 - Math.abs(node.phase - region.phase) * 0.2);
      }
      
      // Function matching score (inferred from node type)
      const nodeFunction = this.inferNodeFunction(node);
      if (nodeFunction === region.function) {
        score += 0.8;
      }
      
      // Performance-based scoring
      if (node.metrics) {
        score += node.metrics.performance * 0.3;
        
        // High-connection nodes prefer central regions
        const connectionRatio = node.metrics.connections / 10;
        if (region.function === 'processing' && connectionRatio > 0.5) {
          score += 0.4;
        }
      }
      
      if (score > bestScore) {
        bestScore = score;
        bestRegion = regionId;
      }
    });
    
    return bestRegion;
  }

  private inferNodeFunction(node: ComponentNode): BrainRegion['function'] {
    switch (node.type) {
      case 'phase':
        return node.phase <= 2 ? 'perception' : 
               node.phase === 3 ? 'memory' :
               node.phase === 4 ? 'control' : 'output';
      case 'engine':
        return 'processing';
      case 'module':
        return 'memory';
      case 'layer':
        return 'control';
      default:
        return 'processing';
    }
  }

  private balanceRegionLoads(assignments: Map<string, ComponentNode[]>): void {
    // Redistribute nodes if regions are over capacity
    const overloadedRegions = Array.from(assignments.entries())
      .filter(([regionId, nodes]) => {
        const region = this.brainRegions.get(regionId);
        return region && nodes.length > region.capacity;
      });
    
    overloadedRegions.forEach(([regionId, nodes]) => {
      const region = this.brainRegions.get(regionId)!;
      const excess = nodes.length - region.capacity;
      const excessNodes = nodes.splice(-excess, excess);
      
      // Redistribute excess nodes to connected regions with capacity
      excessNodes.forEach(node => {
        const alternativeRegion = this.findAlternativeRegion(regionId, assignments);
        if (alternativeRegion) {
          const altNodes = assignments.get(alternativeRegion) || [];
          altNodes.push(node);
          assignments.set(alternativeRegion, altNodes);
        }
      });
    });
  }

  private findAlternativeRegion(
    sourceRegionId: string, 
    assignments: Map<string, ComponentNode[]>
  ): string | null {
    const sourceRegion = this.brainRegions.get(sourceRegionId);
    if (!sourceRegion) return null;
    
    // Try connected regions first
    for (const connectedId of sourceRegion.connections) {
      const connectedRegion = this.brainRegions.get(connectedId);
      const currentNodes = assignments.get(connectedId) || [];
      
      if (connectedRegion && currentNodes.length < connectedRegion.capacity) {
        return connectedId;
      }
    }
    
    // Try any region with capacity
    for (const [regionId, nodes] of assignments.entries()) {
      const region = this.brainRegions.get(regionId);
      if (region && nodes.length < region.capacity) {
        return regionId;
      }
    }
    
    return null;
  }

  private positionNodesInRegions(
    nodes: ComponentNode[],
    assignments: Map<string, ComponentNode[]>
  ): void {
    assignments.forEach((regionNodes, regionId) => {
      const region = this.brainRegions.get(regionId);
      if (!region || regionNodes.length === 0) return;
      
      if (regionNodes.length === 1) {
        // Single node at region center with slight offset
        const offset = this.getRandomOffset(20);
        regionNodes[0].position = {
          x: region.center.x + offset.x,
          y: region.center.y + offset.y,
          z: this.dimensions === '3d' ? (region.center.z || 0) + offset.z : undefined
        };
      } else {
        // Multiple nodes arranged in optimized pattern
        this.arrangeNodesInRegion(regionNodes, region);
      }
    });
  }

  private arrangeNodesInRegion(nodes: ComponentNode[], region: BrainRegion): void {
    const count = nodes.length;
    const centerX = region.center.x;
    const centerY = region.center.y;
    const centerZ = region.center.z || 0;
    const radius = region.radius * 0.8; // Leave some margin
    
    if (count === 2) {
      // Two nodes: position on opposite sides
      nodes[0].position = {
        x: centerX - radius * 0.5,
        y: centerY,
        z: this.dimensions === '3d' ? centerZ : undefined
      };
      nodes[1].position = {
        x: centerX + radius * 0.5,
        y: centerY,
        z: this.dimensions === '3d' ? centerZ : undefined
      };
    } else if (count <= 8) {
      // Few nodes: arrange in circle/sphere
      this.arrangeInCircle(nodes, centerX, centerY, centerZ, radius);
    } else {
      // Many nodes: use spiral pattern
      this.arrangeInSpiral(nodes, centerX, centerY, centerZ, radius);
    }
  }

  private arrangeInCircle(
    nodes: ComponentNode[],
    centerX: number,
    centerY: number,
    centerZ: number,
    radius: number
  ): void {
    const count = nodes.length;
    const angleStep = (2 * Math.PI) / count;
    
    nodes.forEach((node, i) => {
      const angle = i * angleStep;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      
      node.position = {
        x,
        y,
        z: this.dimensions === '3d' ? centerZ + this.getRandomOffset(radius * 0.3).z : undefined
      };
    });
  }

  private arrangeInSpiral(
    nodes: ComponentNode[],
    centerX: number,
    centerY: number,
    centerZ: number,
    maxRadius: number
  ): void {
    const count = nodes.length;
    const a = maxRadius / (2 * Math.PI * Math.sqrt(count));
    
    nodes.forEach((node, i) => {
      const t = Math.sqrt(i / count) * 2 * Math.PI * Math.sqrt(count);
      const r = a * t;
      
      const x = centerX + r * Math.cos(t);
      const y = centerY + r * Math.sin(t);
      
      node.position = {
        x,
        y,
        z: this.dimensions === '3d' ? centerZ + this.getRandomOffset(maxRadius * 0.2).z : undefined
      };
    });
  }

  private applyForceDirectedRefinement(
    nodes: ComponentNode[],
    edges: ComponentEdge[]
  ): void {
    const iterations = 100;
    const coolingFactor = 0.95;
    let temperature = Math.min(this.width, this.height) * 0.1;
    
    for (let iter = 0; iter < iterations; iter++) {
      nodes.forEach(node => {
        if (!node.position) return;
        
        let fx = 0, fy = 0, fz = 0;
        
        // Repulsive forces from other nodes
        nodes.forEach(other => {
          if (node.id === other.id || !other.position) return;
          
          const dx = node.position!.x - other.position.x;
          const dy = node.position!.y - other.position.y;
          const dz = this.dimensions === '3d' && node.position!.z !== undefined && other.position.z !== undefined
            ? node.position!.z - other.position.z : 0;
          
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1; // Avoid division by zero
          const repulsiveForce = this.layoutConstraints.preferredDistance * this.layoutConstraints.preferredDistance / distance;
          
          fx += (dx / distance) * repulsiveForce;
          fy += (dy / distance) * repulsiveForce;
          if (this.dimensions === '3d') fz += (dz / distance) * repulsiveForce;
        });
        
        // Attractive forces from connected nodes
        edges.forEach(edge => {
          let connected: ComponentNode | undefined;
          
          if (edge.source === node.id) {
            connected = nodes.find(n => n.id === edge.target);
          } else if (edge.target === node.id) {
            connected = nodes.find(n => n.id === edge.source);
          }
          
          if (!connected || !connected.position) return;
          
          const dx = connected.position.x - node.position!.x;
          const dy = connected.position.y - node.position!.y;
          const dz = this.dimensions === '3d' && connected.position.z !== undefined && node.position!.z !== undefined
            ? connected.position.z - node.position!.z : 0;
          
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1;
          const attractiveForce = distance * distance / this.layoutConstraints.preferredDistance;
          
          fx += (dx / distance) * attractiveForce;
          fy += (dy / distance) * attractiveForce;
          if (this.dimensions === '3d') fz += (dz / distance) * attractiveForce;
        });
        
        // Apply displacement with temperature cooling
        const displacement = Math.min(temperature, Math.sqrt(fx * fx + fy * fy + fz * fz));
        const totalForce = Math.sqrt(fx * fx + fy * fy + fz * fz) + 1;
        
        node.position.x += (fx / totalForce) * displacement;
        node.position.y += (fy / totalForce) * displacement;
        if (this.dimensions === '3d' && node.position.z !== undefined) {
          node.position.z += (fz / totalForce) * displacement;
        }
        
        // Keep nodes within bounds
        node.position.x = Math.max(50, Math.min(this.width - 50, node.position.x));
        node.position.y = Math.max(50, Math.min(this.height - 50, node.position.y));
        if (this.dimensions === '3d' && node.position.z !== undefined) {
          node.position.z = Math.max(50, Math.min(this.depth - 50, node.position.z));
        }
      });
      
      temperature *= coolingFactor;
    }
  }

  private optimizeLayout(nodes: ComponentNode[], edges: ComponentEdge[]): void {
    // Minimize edge crossings using layer-by-layer optimization
    this.minimizeEdgeCrossings(nodes, edges);
    
    // Reduce node overlaps
    this.resolveNodeOverlaps(nodes);
    
    // Optimize edge lengths
    this.optimizeEdgeLengths(nodes, edges);
  }

  private minimizeEdgeCrossings(nodes: ComponentNode[], edges: ComponentEdge[]): void {
    // Group nodes by phase for hierarchical optimization
    const phaseGroups = new Map<number, ComponentNode[]>();
    
    nodes.forEach(node => {
      if (!phaseGroups.has(node.phase)) {
        phaseGroups.set(node.phase, []);
      }
      phaseGroups.get(node.phase)!.push(node);
    });
    
    // Sort nodes within each phase to minimize crossings
    phaseGroups.forEach((phaseNodes, phase) => {
      if (phaseNodes.length <= 1) return;
      
      const connections = this.getPhaseConnections(phaseNodes, edges);
      const optimizedOrder = this.optimizeNodeOrder(phaseNodes, connections);
      
      // Update positions based on optimized order
      optimizedOrder.forEach((node, index) => {
        if (!node.position) return;
        
        const phaseWidth = this.width / phaseGroups.size;
        const nodeSpacing = phaseWidth / (phaseNodes.length + 1);
        
        node.position.y = (index + 1) * nodeSpacing;
      });
    });
  }

  private getPhaseConnections(nodes: ComponentNode[], edges: ComponentEdge[]): Map<string, string[]> {
    const connections = new Map<string, string[]>();
    
    nodes.forEach(node => {
      connections.set(node.id, []);
    });
    
    edges.forEach(edge => {
      const sourceConnections = connections.get(edge.source);
      const targetConnections = connections.get(edge.target);
      
      if (sourceConnections) sourceConnections.push(edge.target);
      if (targetConnections) targetConnections.push(edge.source);
    });
    
    return connections;
  }

  private optimizeNodeOrder(nodes: ComponentNode[], connections: Map<string, string[]>): ComponentNode[] {
    // Use a simple heuristic: nodes with more connections go to center
    return nodes.sort((a, b) => {
      const aConnections = connections.get(a.id)?.length || 0;
      const bConnections = connections.get(b.id)?.length || 0;
      
      // Secondary sort by performance if available
      const aPerf = a.metrics?.performance || 0;
      const bPerf = b.metrics?.performance || 0;
      
      if (aConnections !== bConnections) {
        return bConnections - aConnections;
      }
      return bPerf - aPerf;
    });
  }

  private resolveNodeOverlaps(nodes: ComponentNode[]): void {
    const minDistance = this.layoutConstraints.minDistance;
    
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const node1 = nodes[i];
        const node2 = nodes[j];
        
        if (!node1.position || !node2.position) continue;
        
        const dx = node2.position.x - node1.position.x;
        const dy = node2.position.y - node1.position.y;
        const dz = this.dimensions === '3d' && node1.position.z !== undefined && node2.position.z !== undefined
          ? node2.position.z - node1.position.z : 0;
        
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        
        if (distance < minDistance) {
          const overlap = minDistance - distance;
          const moveDistance = overlap / 2;
          
          if (distance > 0) {
            const moveX = (dx / distance) * moveDistance;
            const moveY = (dy / distance) * moveDistance;
            const moveZ = this.dimensions === '3d' ? (dz / distance) * moveDistance : 0;
            
            node1.position.x -= moveX;
            node1.position.y -= moveY;
            node2.position.x += moveX;
            node2.position.y += moveY;
            
            if (this.dimensions === '3d' && node1.position.z !== undefined && node2.position.z !== undefined) {
              node1.position.z -= moveZ;
              node2.position.z += moveZ;
            }
          }
        }
      }
    }
  }

  private optimizeEdgeLengths(nodes: ComponentNode[], edges: ComponentEdge[]): void {
    const targetLength = this.layoutConstraints.preferredDistance;
    
    edges.forEach(edge => {
      const sourceNode = nodes.find(n => n.id === edge.source);
      const targetNode = nodes.find(n => n.id === edge.target);
      
      if (!sourceNode?.position || !targetNode?.position) return;
      
      const dx = targetNode.position.x - sourceNode.position.x;
      const dy = targetNode.position.y - sourceNode.position.y;
      const dz = this.dimensions === '3d' && sourceNode.position.z !== undefined && targetNode.position.z !== undefined
        ? targetNode.position.z - sourceNode.position.z : 0;
      
      const currentLength = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      if (Math.abs(currentLength - targetLength) > targetLength * 0.2) {
        const adjustment = (targetLength - currentLength) * 0.1;
        
        if (currentLength > 0) {
          const adjustX = (dx / currentLength) * adjustment;
          const adjustY = (dy / currentLength) * adjustment;
          const adjustZ = this.dimensions === '3d' ? (dz / currentLength) * adjustment : 0;
          
          sourceNode.position.x -= adjustX / 2;
          sourceNode.position.y -= adjustY / 2;
          targetNode.position.x += adjustX / 2;
          targetNode.position.y += adjustY / 2;
          
          if (this.dimensions === '3d' && sourceNode.position.z !== undefined && targetNode.position.z !== undefined) {
            sourceNode.position.z -= adjustZ / 2;
            targetNode.position.z += adjustZ / 2;
          }
        }
      }
    });
  }

  public computeLayoutMetrics(nodes: ComponentNode[], edges: ComponentEdge[]): LayoutMetrics {
    return {
      crossings: this.countEdgeCrossings(nodes, edges),
      totalEdgeLength: this.calculateTotalEdgeLength(nodes, edges),
      nodeOverlap: this.calculateNodeOverlap(nodes),
      hierarchyViolations: this.countHierarchyViolations(nodes, edges),
      clusteringCoefficient: this.calculateClusteringCoefficient(nodes, edges),
      averagePathLength: this.calculateAveragePathLength(nodes, edges)
    };
  }

  private countEdgeCrossings(nodes: ComponentNode[], edges: ComponentEdge[]): number {
    let crossings = 0;
    
    for (let i = 0; i < edges.length; i++) {
      for (let j = i + 1; j < edges.length; j++) {
        const edge1 = edges[i];
        const edge2 = edges[j];
        
        if (this.edgesIntersect(edge1, edge2, nodes)) {
          crossings++;
        }
      }
    }
    
    return crossings;
  }

  private edgesIntersect(edge1: ComponentEdge, edge2: ComponentEdge, nodes: ComponentNode[]): boolean {
    // Check if two edges intersect (2D only for simplicity)
    const node1a = nodes.find(n => n.id === edge1.source)?.position;
    const node1b = nodes.find(n => n.id === edge1.target)?.position;
    const node2a = nodes.find(n => n.id === edge2.source)?.position;
    const node2b = nodes.find(n => n.id === edge2.target)?.position;
    
    if (!node1a || !node1b || !node2a || !node2b) return false;
    
    // Line intersection algorithm
    const d1 = this.orientation(node1a, node1b, node2a);
    const d2 = this.orientation(node1a, node1b, node2b);
    const d3 = this.orientation(node2a, node2b, node1a);
    const d4 = this.orientation(node2a, node2b, node1b);
    
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
      return true;
    }
    
    return false;
  }

  private orientation(p: { x: number; y: number }, q: { x: number; y: number }, r: { x: number; y: number }): number {
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
  }

  private calculateTotalEdgeLength(nodes: ComponentNode[], edges: ComponentEdge[]): number {
    let totalLength = 0;
    
    edges.forEach(edge => {
      const sourcePos = nodes.find(n => n.id === edge.source)?.position;
      const targetPos = nodes.find(n => n.id === edge.target)?.position;
      
      if (sourcePos && targetPos) {
        const dx = targetPos.x - sourcePos.x;
        const dy = targetPos.y - sourcePos.y;
        const dz = this.dimensions === '3d' && sourcePos.z !== undefined && targetPos.z !== undefined
          ? targetPos.z - sourcePos.z : 0;
        
        totalLength += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
    });
    
    return totalLength;
  }

  private calculateNodeOverlap(nodes: ComponentNode[]): number {
    let overlapCount = 0;
    const minDistance = this.layoutConstraints.minDistance;
    
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const pos1 = nodes[i].position;
        const pos2 = nodes[j].position;
        
        if (pos1 && pos2) {
          const dx = pos2.x - pos1.x;
          const dy = pos2.y - pos1.y;
          const dz = this.dimensions === '3d' && pos1.z !== undefined && pos2.z !== undefined
            ? pos2.z - pos1.z : 0;
          
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
          
          if (distance < minDistance) {
            overlapCount++;
          }
        }
      }
    }
    
    return overlapCount;
  }

  private countHierarchyViolations(nodes: ComponentNode[], edges: ComponentEdge[]): number {
    let violations = 0;
    
    edges.forEach(edge => {
      const sourceNode = nodes.find(n => n.id === edge.source);
      const targetNode = nodes.find(n => n.id === edge.target);
      
      if (sourceNode && targetNode && sourceNode.phase > targetNode.phase) {
        violations++;
      }
    });
    
    return violations;
  }

  private calculateClusteringCoefficient(nodes: ComponentNode[], edges: ComponentEdge[]): number {
    // Simplified clustering coefficient calculation
    const adjacencyMap = new Map<string, Set<string>>();
    
    nodes.forEach(node => {
      adjacencyMap.set(node.id, new Set());
    });
    
    edges.forEach(edge => {
      adjacencyMap.get(edge.source)?.add(edge.target);
      adjacencyMap.get(edge.target)?.add(edge.source);
    });
    
    let totalCoefficient = 0;
    let nodeCount = 0;
    
    adjacencyMap.forEach((neighbors, nodeId) => {
      const degree = neighbors.size;
      if (degree < 2) return;
      
      let triangles = 0;
      const neighborsArray = Array.from(neighbors);
      
      for (let i = 0; i < neighborsArray.length; i++) {
        for (let j = i + 1; j < neighborsArray.length; j++) {
          if (adjacencyMap.get(neighborsArray[i])?.has(neighborsArray[j])) {
            triangles++;
          }
        }
      }
      
      const possibleTriangles = (degree * (degree - 1)) / 2;
      totalCoefficient += triangles / possibleTriangles;
      nodeCount++;
    });
    
    return nodeCount > 0 ? totalCoefficient / nodeCount : 0;
  }

  private calculateAveragePathLength(nodes: ComponentNode[], edges: ComponentEdge[]): number {
    // Simplified average path length using BFS
    const adjacencyMap = new Map<string, string[]>();
    
    nodes.forEach(node => {
      adjacencyMap.set(node.id, []);
    });
    
    edges.forEach(edge => {
      adjacencyMap.get(edge.source)?.push(edge.target);
      adjacencyMap.get(edge.target)?.push(edge.source);
    });
    
    let totalPathLength = 0;
    let pathCount = 0;
    
    nodes.forEach(startNode => {
      const distances = this.bfsDistances(startNode.id, adjacencyMap);
      
      Object.values(distances).forEach(distance => {
        if (distance > 0 && distance < Infinity) {
          totalPathLength += distance;
          pathCount++;
        }
      });
    });
    
    return pathCount > 0 ? totalPathLength / pathCount : 0;
  }

  private bfsDistances(startId: string, adjacencyMap: Map<string, string[]>): Record<string, number> {
    const distances: Record<string, number> = {};
    const queue: { id: string; distance: number }[] = [{ id: startId, distance: 0 }];
    const visited = new Set<string>();
    
    while (queue.length > 0) {
      const { id, distance } = queue.shift()!;
      
      if (visited.has(id)) continue;
      visited.add(id);
      distances[id] = distance;
      
      const neighbors = adjacencyMap.get(id) || [];
      neighbors.forEach(neighborId => {
        if (!visited.has(neighborId)) {
          queue.push({ id: neighborId, distance: distance + 1 });
        }
      });
    }
    
    return distances;
  }

  private getRandomOffset(maxOffset: number): { x: number; y: number; z: number } {
    return {
      x: (Math.random() - 0.5) * maxOffset,
      y: (Math.random() - 0.5) * maxOffset,
      z: (Math.random() - 0.5) * maxOffset
    };
  }

  // Public API methods
  public updateConstraints(constraints: Partial<LayoutConstraints>): void {
    this.layoutConstraints = { ...this.layoutConstraints, ...constraints };
  }

  public updateDimensions(width: number, height: number, depth?: number): void {
    this.width = width;
    this.height = height;
    if (depth !== undefined) this.depth = depth;
    
    // Reinitialize brain regions with new dimensions
    this.initializeBrainRegions();
  }

  public getBrainRegions(): Map<string, BrainRegion> {
    return new Map(this.brainRegions);
  }

  public addBrainRegion(region: BrainRegion): void {
    this.brainRegions.set(region.id, region);
  }

  public removeBrainRegion(regionId: string): void {
    this.brainRegions.delete(regionId);
  }
}