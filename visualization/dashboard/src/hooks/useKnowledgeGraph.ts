import { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import { useRealTimeData } from './useRealTimeData';
import { KnowledgeGraphData, KnowledgeNode, KnowledgeEdge, KnowledgeCluster, GraphMetrics } from '../types';

// Knowledge Graph specific interfaces
export interface KnowledgeGraphOptions {
  historySize?: number;
  clusterThreshold?: number;
  edgeWeightThreshold?: number;
  enableCommunityDetection?: boolean;
  trackPathAnalysis?: boolean;
  spatialLayout?: 'force' | 'hierarchical' | 'circular' | 'grid';
  samplingRate?: number;
  maxNodes?: number;
  maxEdges?: number;
}

export interface NodeChangeEvent {
  timestamp: number;
  nodeId: string;
  changeType: 'added' | 'removed' | 'weight_changed' | 'position_changed' | 'metadata_changed';
  oldValue?: any;
  newValue?: any;
  impact: number; // Measure of change impact on overall graph
}

export interface EdgeChangeEvent {
  timestamp: number;
  edgeId: string;
  changeType: 'added' | 'removed' | 'weight_changed' | 'confidence_changed';
  source: string;
  target: string;
  oldValue?: any;
  newValue?: any;
  impact: number;
}

export interface ClusterFormation {
  id: string;
  nodes: string[];
  centroid: { x: number; y: number };
  density: number;
  topic: string;
  coherence: number;
  formationTime: number;
  stability: number;
  growthRate: number;
  influences: string[]; // External factors influencing cluster formation
}

export interface PathOptimization {
  id: string;
  path: string[];
  originalLength: number;
  optimizedLength: number;
  improvement: number;
  timestamp: number;
  pathType: 'shortest' | 'strongest' | 'most_relevant';
  confidence: number;
}

export interface GraphEvolution {
  timestamp: number;
  nodeCount: number;
  edgeCount: number;
  clusterCount: number;
  avgClusterSize: number;
  graphDensity: number;
  modularity: number;
  smallWorldness: number;
  assortativity: number;
  centralNodes: string[];
  bridgeNodes: string[];
  isolatedNodes: string[];
}

export interface SemanticAnalysis {
  conceptDrift: Array<{
    nodeId: string;
    driftVector: { x: number; y: number };
    magnitude: number;
    direction: 'towards' | 'away_from';
    relatedConcepts: string[];
  }>;
  topicEvolution: Array<{
    topic: string;
    strength: number;
    trend: 'emerging' | 'stable' | 'declining';
    keyNodes: string[];
    coherence: number;
  }>;
  semanticClusters: Array<{
    id: string;
    topic: string;
    nodes: string[];
    semanticDistance: number;
    coherence: number;
    representativeTerms: string[];
  }>;
}

export interface KnowledgeGraphMetrics {
  nodeMetrics: {
    totalNodes: number;
    activeNodes: number;
    centralityDistribution: Record<string, number>;
    typeDistribution: Record<string, number>;
    avgConnectivity: number;
  };
  edgeMetrics: {
    totalEdges: number;
    activeEdges: number;
    avgWeight: number;
    avgConfidence: number;
    typeDistribution: Record<string, number>;
  };
  clusterMetrics: {
    totalClusters: number;
    avgClusterSize: number;
    clusterCoherence: number;
    modularityScore: number;
    communityStability: number;
  };
  pathMetrics: {
    avgPathLength: number;
    diameter: number;
    efficiency: number;
    redundancy: number;
  };
  dynamicsMetrics: {
    changeRate: number;
    growthRate: number;
    stabilityScore: number;
    evolutionSpeed: number;
  };
}

// Graph Analysis Engine
class GraphAnalysisEngine {
  private nodeHistory: Map<string, Array<{ timestamp: number; node: KnowledgeNode }>> = new Map();
  private edgeHistory: Map<string, Array<{ timestamp: number; edge: KnowledgeEdge }>> = new Map();
  private communityCache: Map<string, string[]> = new Map(); // node -> community
  private centralityCache: Map<string, number> = new Map();
  private readonly historySize: number;

  constructor(historySize: number = 200) {
    this.historySize = historySize;
  }

  updateGraph(graphData: KnowledgeGraphData): {
    nodeChanges: NodeChangeEvent[];
    edgeChanges: EdgeChangeEvent[];
    newClusters: ClusterFormation[];
    pathOptimizations: PathOptimization[];
  } {
    const timestamp = Date.now();
    const nodeChanges: NodeChangeEvent[] = [];
    const edgeChanges: EdgeChangeEvent[] = [];
    const newClusters: ClusterFormation[] = [];
    const pathOptimizations: PathOptimization[] = [];

    // Process node changes
    graphData.nodes.forEach(node => {
      const history = this.nodeHistory.get(node.id) || [];
      const previousNode = history.length > 0 ? history[history.length - 1].node : null;

      if (!previousNode) {
        // New node
        nodeChanges.push({
          timestamp,
          nodeId: node.id,
          changeType: 'added',
          newValue: node,
          impact: this.calculateNodeImpact(node, graphData),
        });
      } else {
        // Check for changes
        if (Math.abs(node.weight - previousNode.weight) > 0.01) {
          nodeChanges.push({
            timestamp,
            nodeId: node.id,
            changeType: 'weight_changed',
            oldValue: previousNode.weight,
            newValue: node.weight,
            impact: Math.abs(node.weight - previousNode.weight),
          });
        }

        if (node.position.x !== previousNode.position.x || node.position.y !== previousNode.position.y) {
          nodeChanges.push({
            timestamp,
            nodeId: node.id,
            changeType: 'position_changed',
            oldValue: previousNode.position,
            newValue: node.position,
            impact: this.calculatePositionChangeImpact(previousNode.position, node.position),
          });
        }
      }

      // Update history
      history.push({ timestamp, node });
      if (history.length > this.historySize) {
        history.shift();
      }
      this.nodeHistory.set(node.id, history);
    });

    // Process edge changes
    graphData.edges.forEach(edge => {
      const history = this.edgeHistory.get(edge.id) || [];
      const previousEdge = history.length > 0 ? history[history.length - 1].edge : null;

      if (!previousEdge) {
        // New edge
        edgeChanges.push({
          timestamp,
          edgeId: edge.id,
          changeType: 'added',
          source: edge.source,
          target: edge.target,
          newValue: edge,
          impact: edge.weight * edge.confidence,
        });
      } else {
        // Check for changes
        if (Math.abs(edge.weight - previousEdge.weight) > 0.01) {
          edgeChanges.push({
            timestamp,
            edgeId: edge.id,
            changeType: 'weight_changed',
            source: edge.source,
            target: edge.target,
            oldValue: previousEdge.weight,
            newValue: edge.weight,
            impact: Math.abs(edge.weight - previousEdge.weight),
          });
        }

        if (Math.abs(edge.confidence - previousEdge.confidence) > 0.01) {
          edgeChanges.push({
            timestamp,
            edgeId: edge.id,
            changeType: 'confidence_changed',
            source: edge.source,
            target: edge.target,
            oldValue: previousEdge.confidence,
            newValue: edge.confidence,
            impact: Math.abs(edge.confidence - previousEdge.confidence),
          });
        }
      }

      // Update history
      history.push({ timestamp, edge });
      if (history.length > this.historySize) {
        history.shift();
      }
      this.edgeHistory.set(edge.id, history);
    });

    // Detect new cluster formations
    const currentClusters = this.detectCommunities(graphData);
    currentClusters.forEach(cluster => {
      if (this.isNewCluster(cluster)) {
        const formation = this.analyzeClusterFormation(cluster, graphData);
        newClusters.push(formation);
      }
    });

    // Find path optimizations
    const optimizations = this.findPathOptimizations(graphData);
    pathOptimizations.push(...optimizations);

    return { nodeChanges, edgeChanges, newClusters, pathOptimizations };
  }

  private calculateNodeImpact(node: KnowledgeNode, graph: KnowledgeGraphData): number {
    // Calculate impact based on centrality and connections
    const connections = graph.edges.filter(e => e.source === node.id || e.target === node.id);
    const degree = connections.length;
    const weightedDegree = connections.reduce((sum, e) => sum + e.weight, 0);
    
    return (degree * node.weight + weightedDegree) / (graph.nodes.length + 1);
  }

  private calculatePositionChangeImpact(oldPos: { x: number; y: number }, newPos: { x: number; y: number }): number {
    const dx = newPos.x - oldPos.x;
    const dy = newPos.y - oldPos.y;
    return Math.sqrt(dx * dx + dy * dy) / 100; // Normalize by arbitrary scale
  }

  detectCommunities(graph: KnowledgeGraphData): Array<{ id: string; nodes: string[]; density: number }> {
    // Simplified community detection using connected components
    const visited = new Set<string>();
    const communities: Array<{ id: string; nodes: string[]; density: number }> = [];

    const adjacencyList = new Map<string, string[]>();
    
    // Build adjacency list
    graph.nodes.forEach(node => adjacencyList.set(node.id, []));
    graph.edges.forEach(edge => {
      if (edge.weight > 0.3) { // Threshold for meaningful connections
        adjacencyList.get(edge.source)?.push(edge.target);
        adjacencyList.get(edge.target)?.push(edge.source);
      }
    });

    // DFS to find connected components
    const dfs = (nodeId: string, component: string[]): void => {
      visited.add(nodeId);
      component.push(nodeId);
      
      const neighbors = adjacencyList.get(nodeId) || [];
      neighbors.forEach(neighbor => {
        if (!visited.has(neighbor)) {
          dfs(neighbor, component);
        }
      });
    };

    let communityId = 0;
    graph.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        const component: string[] = [];
        dfs(node.id, component);
        
        if (component.length >= 2) { // Minimum community size
          const density = this.calculateCommunityDensity(component, graph);
          communities.push({
            id: `community_${communityId++}`,
            nodes: component,
            density,
          });
        }
      }
    });

    return communities;
  }

  private calculateCommunityDensity(nodes: string[], graph: KnowledgeGraphData): number {
    const communityEdges = graph.edges.filter(e => 
      nodes.includes(e.source) && nodes.includes(e.target)
    );
    
    const maxPossibleEdges = nodes.length * (nodes.length - 1) / 2;
    return maxPossibleEdges > 0 ? communityEdges.length / maxPossibleEdges : 0;
  }

  private isNewCluster(cluster: { id: string; nodes: string[]; density: number }): boolean {
    // Check if this cluster configuration is significantly different from cached ones
    const nodeSet = new Set(cluster.nodes);
    
    for (const [cachedId, cachedNodes] of this.communityCache.entries()) {
      const overlap = cachedNodes.filter(node => nodeSet.has(node)).length;
      const similarity = overlap / Math.max(cluster.nodes.length, cachedNodes.length);
      
      if (similarity > 0.8) {
        return false; // Too similar to existing cluster
      }
    }

    // Cache this cluster
    this.communityCache.set(cluster.id, cluster.nodes);
    return true;
  }

  private analyzeClusterFormation(
    cluster: { id: string; nodes: string[]; density: number },
    graph: KnowledgeGraphData
  ): ClusterFormation {
    const timestamp = Date.now();
    const clusterNodes = graph.nodes.filter(n => cluster.nodes.includes(n.id));
    
    // Calculate centroid
    const centroid = clusterNodes.reduce(
      (acc, node) => ({
        x: acc.x + node.position.x,
        y: acc.y + node.position.y,
      }),
      { x: 0, y: 0 }
    );
    centroid.x /= clusterNodes.length;
    centroid.y /= clusterNodes.length;

    // Infer topic from node types and metadata
    const topicCounts = clusterNodes.reduce((counts, node) => {
      const topic = node.metadata?.topic || node.type || 'unknown';
      counts[topic] = (counts[topic] || 0) + 1;
      return counts;
    }, {} as Record<string, number>);
    
    const topic = Object.entries(topicCounts).reduce((a, b) => 
      topicCounts[a[0]] > topicCounts[b[0]] ? a : b
    )[0];

    // Calculate coherence
    const coherence = this.calculateClusterCoherence(cluster.nodes, graph);
    
    // Estimate growth rate based on recent node additions
    const growthRate = this.calculateGrowthRate(cluster.nodes);

    return {
      id: cluster.id,
      nodes: cluster.nodes,
      centroid,
      density: cluster.density,
      topic,
      coherence,
      formationTime: timestamp,
      stability: coherence * cluster.density,
      growthRate,
      influences: [], // Would need more sophisticated analysis
    };
  }

  private calculateClusterCoherence(nodes: string[], graph: KnowledgeGraphData): number {
    if (nodes.length < 2) return 1.0;

    const clusterEdges = graph.edges.filter(e => 
      nodes.includes(e.source) && nodes.includes(e.target)
    );
    
    const avgWeight = clusterEdges.reduce((sum, e) => sum + e.weight, 0) / Math.max(clusterEdges.length, 1);
    const avgConfidence = clusterEdges.reduce((sum, e) => sum + e.confidence, 0) / Math.max(clusterEdges.length, 1);
    
    return (avgWeight + avgConfidence) / 2;
  }

  private calculateGrowthRate(nodes: string[]): number {
    // Calculate growth rate based on node addition timestamps
    // This is simplified - in practice you'd track actual timestamps
    let recentAdditions = 0;
    const recentThreshold = Date.now() - 5000; // 5 seconds

    nodes.forEach(nodeId => {
      const history = this.nodeHistory.get(nodeId);
      if (history && history.length > 0) {
        const firstEntry = history[0];
        if (firstEntry.timestamp > recentThreshold) {
          recentAdditions++;
        }
      }
    });

    return recentAdditions / 5; // additions per second
  }

  findPathOptimizations(graph: KnowledgeGraphData): PathOptimization[] {
    const optimizations: PathOptimization[] = [];
    
    // Find high-traffic node pairs that might benefit from optimization
    const importantNodes = graph.nodes
      .sort((a, b) => b.weight - a.weight)
      .slice(0, Math.min(10, graph.nodes.length)); // Top 10 nodes

    for (let i = 0; i < importantNodes.length; i++) {
      for (let j = i + 1; j < importantNodes.length; j++) {
        const path = this.findShortestPath(importantNodes[i].id, importantNodes[j].id, graph);
        if (path.length > 3) { // Only optimize paths longer than 3 nodes
          const optimizedPath = this.optimizePath(path, graph);
          if (optimizedPath.length < path.length) {
            optimizations.push({
              id: `opt_${importantNodes[i].id}_${importantNodes[j].id}`,
              path: optimizedPath,
              originalLength: path.length,
              optimizedLength: optimizedPath.length,
              improvement: (path.length - optimizedPath.length) / path.length,
              timestamp: Date.now(),
              pathType: 'shortest',
              confidence: 0.8, // Would be calculated based on edge confidences
            });
          }
        }
      }
    }

    return optimizations;
  }

  private findShortestPath(startId: string, endId: string, graph: KnowledgeGraphData): string[] {
    // Simplified Dijkstra's algorithm
    const distances = new Map<string, number>();
    const previous = new Map<string, string | null>();
    const unvisited = new Set(graph.nodes.map(n => n.id));

    // Initialize distances
    graph.nodes.forEach(node => {
      distances.set(node.id, node.id === startId ? 0 : Infinity);
      previous.set(node.id, null);
    });

    while (unvisited.size > 0) {
      // Find unvisited node with minimum distance
      let current = '';
      let minDistance = Infinity;
      for (const nodeId of unvisited) {
        const distance = distances.get(nodeId) || Infinity;
        if (distance < minDistance) {
          minDistance = distance;
          current = nodeId;
        }
      }

      if (current === '' || minDistance === Infinity) break;
      if (current === endId) break;

      unvisited.delete(current);

      // Update distances to neighbors
      const neighbors = graph.edges.filter(e => e.source === current || e.target === current);
      for (const edge of neighbors) {
        const neighbor = edge.source === current ? edge.target : edge.source;
        if (!unvisited.has(neighbor)) continue;

        const newDistance = (distances.get(current) || 0) + (1 / Math.max(edge.weight, 0.1));
        if (newDistance < (distances.get(neighbor) || Infinity)) {
          distances.set(neighbor, newDistance);
          previous.set(neighbor, current);
        }
      }
    }

    // Reconstruct path
    const path: string[] = [];
    let current: string | null = endId;
    while (current !== null) {
      path.unshift(current);
      current = previous.get(current) || null;
    }

    return path[0] === startId ? path : []; // Return empty if no path found
  }

  private optimizePath(path: string[], graph: KnowledgeGraphData): string[] {
    // Simple path optimization - remove unnecessary intermediate nodes
    if (path.length <= 2) return path;

    const optimized = [path[0]];
    
    for (let i = 1; i < path.length - 1; i++) {
      const prev = optimized[optimized.length - 1];
      const current = path[i];
      const next = path[i + 1];

      // Check if we can skip current node (direct connection exists)
      const directConnection = graph.edges.find(e => 
        (e.source === prev && e.target === next) || 
        (e.source === next && e.target === prev)
      );

      if (!directConnection || directConnection.weight < 0.3) {
        optimized.push(current); // Keep the intermediate node
      }
    }

    optimized.push(path[path.length - 1]);
    return optimized;
  }

  calculateGraphMetrics(graph: KnowledgeGraphData): GraphMetrics {
    // This extends the basic GraphMetrics with more detailed analysis
    const nodeCount = graph.nodes.length;
    const edgeCount = graph.edges.length;
    const clusterCount = graph.clusters.length;
    
    const density = nodeCount > 1 ? edgeCount / (nodeCount * (nodeCount - 1) / 2) : 0;
    const avgDegree = nodeCount > 0 ? (2 * edgeCount) / nodeCount : 0;

    return {
      nodeCount,
      edgeCount,
      clusterCount,
      density,
      avgDegree,
    };
  }
}

// Main Hook Implementation
export const useKnowledgeGraph = (options: KnowledgeGraphOptions = {}) => {
  const {
    historySize = 200,
    clusterThreshold = 0.3,
    edgeWeightThreshold = 0.1,
    enableCommunityDetection = true,
    trackPathAnalysis = true,
    spatialLayout = 'force',
    samplingRate = 20, // 20Hz for knowledge graph updates
    maxNodes = 1000,
    maxEdges = 5000,
  } = options;

  // Use base real-time data hook
  const {
    data: rawGraphData,
    latest: latestGraph,
    isConnected,
    metrics: baseMetrics,
    aggregatedData,
  } = useRealTimeData<KnowledgeGraphData>('knowledgeGraph', {
    bufferSize: historySize,
    samplingRate,
    aggregationWindow: 10000, // 10 second aggregation window
    enablePerformanceMonitoring: true,
  });

  // Analysis engine
  const [graphEngine] = useState(() => new GraphAnalysisEngine(historySize));

  // State management
  const [nodeChanges, setNodeChanges] = useState<NodeChangeEvent[]>([]);
  const [edgeChanges, setEdgeChanges] = useState<EdgeChangeEvent[]>([]);
  const [clusterFormations, setClusterFormations] = useState<ClusterFormation[]>([]);
  const [pathOptimizations, setPathOptimizations] = useState<PathOptimization[]>([]);
  const [graphEvolution, setGraphEvolution] = useState<GraphEvolution[]>([]);
  const [semanticAnalysis, setSemanticAnalysis] = useState<SemanticAnalysis>({
    conceptDrift: [],
    topicEvolution: [],
    semanticClusters: [],
  });

  // Process knowledge graph updates
  useEffect(() => {
    if (!latestGraph || !isConnected) return;

    try {
      const { nodeChanges: newNodeChanges, edgeChanges: newEdgeChanges, newClusters, pathOptimizations: newOptimizations } = 
        graphEngine.updateGraph(latestGraph);

      // Update state with new changes
      setNodeChanges(prev => {
        const updated = [...prev, ...newNodeChanges];
        return updated.slice(-historySize);
      });

      setEdgeChanges(prev => {
        const updated = [...prev, ...newEdgeChanges];
        return updated.slice(-historySize);
      });

      setClusterFormations(prev => {
        const updated = [...prev, ...newClusters];
        return updated.slice(-50); // Keep last 50 cluster formations
      });

      setPathOptimizations(prev => {
        const updated = [...prev, ...newOptimizations];
        return updated.slice(-100); // Keep last 100 optimizations
      });

      // Update graph evolution
      const evolution: GraphEvolution = {
        timestamp: Date.now(),
        nodeCount: latestGraph.nodes.length,
        edgeCount: latestGraph.edges.length,
        clusterCount: latestGraph.clusters.length,
        avgClusterSize: latestGraph.clusters.length > 0 
          ? latestGraph.clusters.reduce((sum, c) => sum + c.nodes.length, 0) / latestGraph.clusters.length 
          : 0,
        graphDensity: latestGraph.metrics.density,
        modularity: calculateModularity(latestGraph),
        smallWorldness: calculateSmallWorldness(latestGraph),
        assortativity: calculateAssortativity(latestGraph),
        centralNodes: findCentralNodes(latestGraph, 5),
        bridgeNodes: findBridgeNodes(latestGraph, 5),
        isolatedNodes: findIsolatedNodes(latestGraph),
      };

      setGraphEvolution(prev => {
        const updated = [...prev, evolution];
        return updated.slice(-historySize);
      });

      // Update semantic analysis
      if (enableCommunityDetection) {
        const semantic = analyzeSemantics(latestGraph, rawGraphData.slice(-10));
        setSemanticAnalysis(semantic);
      }

    } catch (error) {
      console.error('Error processing knowledge graph updates:', error);
    }
  }, [latestGraph, isConnected, graphEngine, historySize, enableCommunityDetection, rawGraphData]);

  // Calculate comprehensive metrics
  const knowledgeGraphMetrics = useMemo((): KnowledgeGraphMetrics => {
    if (!latestGraph) {
      return {
        nodeMetrics: {
          totalNodes: 0,
          activeNodes: 0,
          centralityDistribution: {},
          typeDistribution: {},
          avgConnectivity: 0,
        },
        edgeMetrics: {
          totalEdges: 0,
          activeEdges: 0,
          avgWeight: 0,
          avgConfidence: 0,
          typeDistribution: {},
        },
        clusterMetrics: {
          totalClusters: 0,
          avgClusterSize: 0,
          clusterCoherence: 0,
          modularityScore: 0,
          communityStability: 0,
        },
        pathMetrics: {
          avgPathLength: 0,
          diameter: 0,
          efficiency: 0,
          redundancy: 0,
        },
        dynamicsMetrics: {
          changeRate: 0,
          growthRate: 0,
          stabilityScore: 0,
          evolutionSpeed: 0,
        },
      };
    }

    const { nodes, edges, clusters } = latestGraph;

    // Node metrics
    const activeNodes = nodes.filter(n => n.weight > 0.1).length;
    const typeDistribution = nodes.reduce((dist, node) => {
      dist[node.type] = (dist[node.type] || 0) + 1;
      return dist;
    }, {} as Record<string, number>);

    const totalDegree = nodes.reduce((sum, node) => {
      const degree = edges.filter(e => e.source === node.id || e.target === node.id).length;
      return sum + degree;
    }, 0);
    const avgConnectivity = nodes.length > 0 ? totalDegree / nodes.length : 0;

    // Edge metrics
    const activeEdges = edges.filter(e => e.weight > edgeWeightThreshold).length;
    const avgWeight = edges.reduce((sum, e) => sum + e.weight, 0) / Math.max(edges.length, 1);
    const avgConfidence = edges.reduce((sum, e) => sum + e.confidence, 0) / Math.max(edges.length, 1);
    const edgeTypeDistribution = edges.reduce((dist, edge) => {
      dist[edge.type] = (dist[edge.type] || 0) + 1;
      return dist;
    }, {} as Record<string, number>);

    // Cluster metrics
    const avgClusterSize = clusters.reduce((sum, c) => sum + c.nodes.length, 0) / Math.max(clusters.length, 1);
    const clusterCoherence = clusters.reduce((sum, c) => sum + c.density, 0) / Math.max(clusters.length, 1);
    const modularityScore = calculateModularity(latestGraph);

    // Path metrics
    const pathMetrics = calculatePathMetrics(latestGraph);

    // Dynamics metrics
    const recentChanges = [...nodeChanges, ...edgeChanges].filter(
      change => Date.now() - change.timestamp <= 10000 // Last 10 seconds
    );
    const changeRate = recentChanges.length / 10; // Changes per second

    const recentEvolution = graphEvolution.slice(-10);
    const growthRate = recentEvolution.length >= 2 
      ? (recentEvolution[recentEvolution.length - 1].nodeCount - recentEvolution[0].nodeCount) / recentEvolution.length
      : 0;

    const stabilityScore = calculateStabilityScore(recentEvolution);
    const evolutionSpeed = calculateEvolutionSpeed(recentEvolution);

    return {
      nodeMetrics: {
        totalNodes: nodes.length,
        activeNodes,
        centralityDistribution: {}, // Would need centrality calculation
        typeDistribution,
        avgConnectivity,
      },
      edgeMetrics: {
        totalEdges: edges.length,
        activeEdges,
        avgWeight,
        avgConfidence,
        typeDistribution: edgeTypeDistribution,
      },
      clusterMetrics: {
        totalClusters: clusters.length,
        avgClusterSize,
        clusterCoherence,
        modularityScore,
        communityStability: 0.8, // Simplified
      },
      pathMetrics,
      dynamicsMetrics: {
        changeRate,
        growthRate,
        stabilityScore,
        evolutionSpeed,
      },
    };
  }, [latestGraph, edgeWeightThreshold, nodeChanges, edgeChanges, graphEvolution]);

  // Cleanup
  useEffect(() => {
    return () => {
      // Cleanup if needed
    };
  }, []);

  return {
    // Core data
    nodes: latestGraph?.nodes || [],
    edges: latestGraph?.edges || [],
    clusters: latestGraph?.clusters || [],
    
    // Change tracking
    nodeChanges,
    edgeChanges,
    clusterFormations,
    pathOptimizations,
    graphEvolution,
    semanticAnalysis,
    
    // Metrics and analysis
    metrics: knowledgeGraphMetrics,
    isConnected,
    
    // Aggregated and historical data
    historicalData: rawGraphData,
    aggregatedData,
    
    // Control methods
    clearHistory: useCallback(() => {
      setNodeChanges([]);
      setEdgeChanges([]);
      setClusterFormations([]);
      setPathOptimizations([]);
      setGraphEvolution([]);
    }, []),
    
    findPath: useCallback((startId: string, endId: string) => {
      if (!latestGraph) return [];
      return graphEngine.findShortestPath(startId, endId, latestGraph);
    }, [latestGraph, graphEngine]),
    
    getNodeNeighbors: useCallback((nodeId: string) => {
      if (!latestGraph) return [];
      return latestGraph.edges
        .filter(e => e.source === nodeId || e.target === nodeId)
        .map(e => e.source === nodeId ? e.target : e.source);
    }, [latestGraph]),
    
    getCommunities: useCallback(() => {
      if (!latestGraph) return [];
      return graphEngine.detectCommunities(latestGraph);
    }, [latestGraph, graphEngine]),
  };
};

// Helper functions
function calculateModularity(graph: KnowledgeGraphData): number {
  // Simplified modularity calculation
  const totalEdges = graph.edges.length;
  if (totalEdges === 0) return 0;

  const clusterEdges = graph.clusters.reduce((sum, cluster) => {
    const clusterNodes = new Set(cluster.nodes);
    const internalEdges = graph.edges.filter(e => 
      clusterNodes.has(e.source) && clusterNodes.has(e.target)
    ).length;
    return sum + internalEdges;
  }, 0);

  return totalEdges > 0 ? clusterEdges / totalEdges : 0;
}

function calculateSmallWorldness(graph: KnowledgeGraphData): number {
  // Simplified small-world calculation
  const avgDegree = graph.nodes.length > 0 ? (2 * graph.edges.length) / graph.nodes.length : 0;
  const clusteringCoeff = graph.clusters.reduce((sum, c) => sum + c.density, 0) / Math.max(graph.clusters.length, 1);
  
  // Simplified characteristic path length
  const pathLength = graph.nodes.length > 1 ? Math.log(graph.nodes.length) / Math.log(avgDegree || 1) : 1;
  
  return pathLength > 0 ? clusteringCoeff / pathLength : 0;
}

function calculateAssortativity(graph: KnowledgeGraphData): number {
  // Simplified assortativity calculation based on node types
  if (graph.edges.length === 0) return 0;

  const nodeTypes = new Map(graph.nodes.map(n => [n.id, n.type]));
  const sameTypeConnections = graph.edges.filter(e => 
    nodeTypes.get(e.source) === nodeTypes.get(e.target)
  ).length;

  return sameTypeConnections / graph.edges.length;
}

function findCentralNodes(graph: KnowledgeGraphData, count: number): string[] {
  // Find nodes with highest degree centrality
  const nodeDegrees = graph.nodes.map(node => ({
    id: node.id,
    degree: graph.edges.filter(e => e.source === node.id || e.target === node.id).length,
  }));

  return nodeDegrees
    .sort((a, b) => b.degree - a.degree)
    .slice(0, count)
    .map(n => n.id);
}

function findBridgeNodes(graph: KnowledgeGraphData, count: number): string[] {
  // Find nodes that connect different clusters
  const bridgeScores = graph.nodes.map(node => {
    const neighbors = graph.edges
      .filter(e => e.source === node.id || e.target === node.id)
      .map(e => e.source === node.id ? e.target : e.source);

    // Count how many different clusters the neighbors belong to
    const clusterSet = new Set<string>();
    neighbors.forEach(neighbor => {
      const cluster = graph.clusters.find(c => c.nodes.includes(neighbor));
      if (cluster) clusterSet.add(cluster.id);
    });

    return { id: node.id, bridgeScore: clusterSet.size };
  });

  return bridgeScores
    .sort((a, b) => b.bridgeScore - a.bridgeScore)
    .slice(0, count)
    .map(n => n.id);
}

function findIsolatedNodes(graph: KnowledgeGraphData): string[] {
  const connectedNodes = new Set<string>();
  graph.edges.forEach(e => {
    connectedNodes.add(e.source);
    connectedNodes.add(e.target);
  });

  return graph.nodes
    .filter(node => !connectedNodes.has(node.id))
    .map(node => node.id);
}

function analyzeSemantics(currentGraph: KnowledgeGraphData, historicalData: KnowledgeGraphData[]): SemanticAnalysis {
  // Simplified semantic analysis
  const conceptDrift = analyzeConcepeDrift(currentGraph, historicalData);
  const topicEvolution = analyzeTopicEvolution(currentGraph, historicalData);
  const semanticClusters = findSemanticClusters(currentGraph);

  return {
    conceptDrift,
    topicEvolution,
    semanticClusters,
  };
}

function analyzeConcepeDrift(currentGraph: KnowledgeGraphData, historicalData: KnowledgeGraphData[]): SemanticAnalysis['conceptDrift'] {
  if (historicalData.length < 2) return [];

  const drift: SemanticAnalysis['conceptDrift'] = [];
  const previousGraph = historicalData[historicalData.length - 2];

  currentGraph.nodes.forEach(currentNode => {
    const previousNode = previousGraph.nodes.find(n => n.id === currentNode.id);
    if (previousNode) {
      const dx = currentNode.position.x - previousNode.position.x;
      const dy = currentNode.position.y - previousNode.position.y;
      const magnitude = Math.sqrt(dx * dx + dy * dy);

      if (magnitude > 5) { // Threshold for significant drift
        drift.push({
          nodeId: currentNode.id,
          driftVector: { x: dx, y: dy },
          magnitude,
          direction: magnitude > 0 ? 'towards' : 'away_from',
          relatedConcepts: [], // Would need more sophisticated analysis
        });
      }
    }
  });

  return drift;
}

function analyzeTopicEvolution(currentGraph: KnowledgeGraphData, historicalData: KnowledgeGraphData[]): SemanticAnalysis['topicEvolution'] {
  const topicStrengths = new Map<string, number>();
  
  currentGraph.clusters.forEach(cluster => {
    const strength = cluster.density * cluster.nodes.length;
    topicStrengths.set(cluster.topic, (topicStrengths.get(cluster.topic) || 0) + strength);
  });

  return Array.from(topicStrengths.entries()).map(([topic, strength]) => ({
    topic,
    strength,
    trend: 'stable' as const, // Would need historical comparison
    keyNodes: currentGraph.clusters
      .filter(c => c.topic === topic)
      .flatMap(c => c.nodes)
      .slice(0, 5),
    coherence: currentGraph.clusters
      .filter(c => c.topic === topic)
      .reduce((sum, c) => sum + c.density, 0) / Math.max(
        currentGraph.clusters.filter(c => c.topic === topic).length, 1
      ),
  }));
}

function findSemanticClusters(graph: KnowledgeGraphData): SemanticAnalysis['semanticClusters'] {
  return graph.clusters.map(cluster => ({
    id: cluster.id,
    topic: cluster.topic,
    nodes: cluster.nodes,
    semanticDistance: 1 / Math.max(cluster.density, 0.01),
    coherence: cluster.density,
    representativeTerms: [], // Would need NLP analysis
  }));
}

function calculatePathMetrics(graph: KnowledgeGraphData): KnowledgeGraphMetrics['pathMetrics'] {
  // Simplified path metrics calculation
  const nodeCount = graph.nodes.length;
  const edgeCount = graph.edges.length;
  
  // Estimate average path length (simplified)
  const avgPathLength = nodeCount > 1 ? Math.log(nodeCount) / Math.log(2) : 0;
  
  // Estimate diameter (simplified)
  const diameter = avgPathLength * 1.5;
  
  // Network efficiency
  const efficiency = nodeCount > 1 ? 1 / avgPathLength : 1;
  
  // Path redundancy
  const redundancy = edgeCount > nodeCount - 1 ? (edgeCount - (nodeCount - 1)) / edgeCount : 0;

  return {
    avgPathLength,
    diameter,
    efficiency,
    redundancy,
  };
}

function calculateStabilityScore(evolution: GraphEvolution[]): number {
  if (evolution.length < 2) return 1.0;

  const variations = evolution.slice(1).map((current, i) => {
    const previous = evolution[i];
    const nodeChange = Math.abs(current.nodeCount - previous.nodeCount) / Math.max(previous.nodeCount, 1);
    const edgeChange = Math.abs(current.edgeCount - previous.edgeCount) / Math.max(previous.edgeCount, 1);
    const clusterChange = Math.abs(current.clusterCount - previous.clusterCount) / Math.max(previous.clusterCount, 1);
    
    return (nodeChange + edgeChange + clusterChange) / 3;
  });

  const avgVariation = variations.reduce((sum, v) => sum + v, 0) / variations.length;
  return 1 / (1 + avgVariation);
}

function calculateEvolutionSpeed(evolution: GraphEvolution[]): number {
  if (evolution.length < 2) return 0;

  const timespan = evolution[evolution.length - 1].timestamp - evolution[0].timestamp;
  const changes = evolution.slice(1).reduce((sum, current, i) => {
    const previous = evolution[i];
    return sum + Math.abs(current.nodeCount - previous.nodeCount) + 
                 Math.abs(current.edgeCount - previous.edgeCount) +
                 Math.abs(current.clusterCount - previous.clusterCount);
  }, 0);

  return timespan > 0 ? (changes * 1000) / timespan : 0; // Changes per second
}

// Export types
export type {
  KnowledgeGraphOptions,
  NodeChangeEvent,
  EdgeChangeEvent,
  ClusterFormation,
  PathOptimization,
  GraphEvolution,
  SemanticAnalysis,
  KnowledgeGraphMetrics,
};