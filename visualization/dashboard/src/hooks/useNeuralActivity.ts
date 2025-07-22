import { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import { useRealTimeData } from './useRealTimeData';
import { NeuralData, NeuralActivity, NeuralLayer, NeuralConnection } from '../types';

// Neural activity specific interfaces
export interface NeuralActivityOptions {
  historySize?: number;
  layerFocusDepth?: number;
  connectionThreshold?: number;
  enableSpikeDetection?: boolean;
  activityWindowMs?: number;
  spatialResolution?: number;
  temporalSmoothing?: number;
  samplingRate?: number;
}

export interface SpikeEvent {
  timestamp: number;
  nodeId: string;
  amplitude: number;
  duration: number;
  layer: number;
  position: { x: number; y: number; z?: number };
  presynapticNodes: string[];
  postsynapticNodes: string[];
}

export interface LayerAnalysis {
  id: string;
  name: string;
  currentActivity: number;
  averageActivity: number;
  peakActivity: number;
  activityTrend: 'rising' | 'falling' | 'stable';
  nodeCount: number;
  activeNodes: number;
  utilization: number;
  spikeRate: number;
  synchrony: number;
  spatialClusters: SpatialCluster[];
  temporalPattern: 'burst' | 'tonic' | 'oscillatory' | 'sparse';
}

export interface SpatialCluster {
  id: string;
  nodes: string[];
  centroid: { x: number; y: number; z?: number };
  radius: number;
  density: number;
  averageActivity: number;
  coherence: number;
}

export interface ConnectivityMatrix {
  layerPairs: Array<{
    fromLayer: string;
    toLayer: string;
    connections: Array<{
      strength: number;
      active: boolean;
      weight: number;
      delay: number;
    }>;
    averageWeight: number;
    activeRatio: number;
    synchrony: number;
  }>;
  overallConnectivity: number;
  networkDensity: number;
  smallWorldIndex: number;
  modularity: number;
}

export interface ActivityWave {
  id: string;
  origin: { x: number; y: number; z?: number };
  propagationSpeed: number;
  amplitude: number;
  wavelength: number;
  direction: { x: number; y: number; z?: number };
  affectedNodes: string[];
  timestamp: number;
  duration: number;
}

export interface NeuralMetrics {
  overallActivity: number;
  layerActivation: Record<string, number>;
  connectionStrength: number;
  networkEfficiency: number;
  spikeFrequency: number;
  spatialCoherence: number;
  temporalStability: number;
  plasticity: number;
  energyConsumption: number;
}

// Spatial Analysis Engine
class SpatialAnalysisEngine {
  private readonly resolution: number;
  private spatialGrid: Map<string, string[]> = new Map(); // Grid cell to node IDs
  private nodePositions: Map<string, { x: number; y: number; z?: number }> = new Map();

  constructor(resolution: number = 10) {
    this.resolution = resolution;
  }

  updateNodePositions(activities: NeuralActivity[]): void {
    this.spatialGrid.clear();
    this.nodePositions.clear();

    activities.forEach(activity => {
      this.nodePositions.set(activity.nodeId, activity.position);
      
      const gridKey = this.getGridKey(activity.position);
      const nodes = this.spatialGrid.get(gridKey) || [];
      nodes.push(activity.nodeId);
      this.spatialGrid.set(gridKey, nodes);
    });
  }

  private getGridKey(position: { x: number; y: number; z?: number }): string {
    const x = Math.floor(position.x / this.resolution);
    const y = Math.floor(position.y / this.resolution);
    const z = position.z !== undefined ? Math.floor(position.z / this.resolution) : 0;
    return `${x},${y},${z}`;
  }

  findSpatialClusters(activities: NeuralActivity[], minClusterSize: number = 3): SpatialCluster[] {
    const clusters: SpatialCluster[] = [];
    const processed = new Set<string>();

    for (const activity of activities) {
      if (processed.has(activity.nodeId) || activity.activation < 0.3) continue;

      const cluster = this.growCluster(activity, activities, processed);
      if (cluster.nodes.length >= minClusterSize) {
        clusters.push(cluster);
      }
    }

    return clusters.sort((a, b) => b.averageActivity - a.averageActivity);
  }

  private growCluster(
    seed: NeuralActivity,
    allActivities: NeuralActivity[],
    processed: Set<string>
  ): SpatialCluster {
    const clusterNodes: NeuralActivity[] = [seed];
    const queue: NeuralActivity[] = [seed];
    processed.add(seed.nodeId);

    const proximityThreshold = this.resolution * 2;

    while (queue.length > 0) {
      const current = queue.shift()!;
      
      for (const activity of allActivities) {
        if (processed.has(activity.nodeId) || activity.activation < 0.2) continue;
        
        const distance = this.calculateDistance(current.position, activity.position);
        if (distance <= proximityThreshold) {
          clusterNodes.push(activity);
          queue.push(activity);
          processed.add(activity.nodeId);
        }
      }
    }

    return this.createClusterFromNodes(clusterNodes);
  }

  private createClusterFromNodes(nodes: NeuralActivity[]): SpatialCluster {
    const centroid = this.calculateCentroid(nodes.map(n => n.position));
    const maxDistance = Math.max(...nodes.map(n => 
      this.calculateDistance(n.position, centroid)
    ));
    
    const averageActivity = nodes.reduce((sum, n) => sum + n.activation, 0) / nodes.length;
    const density = nodes.length / (Math.PI * maxDistance * maxDistance);
    const coherence = this.calculateSpatialCoherence(nodes);

    return {
      id: `cluster_${Date.now()}_${nodes[0].nodeId}`,
      nodes: nodes.map(n => n.nodeId),
      centroid,
      radius: maxDistance,
      density,
      averageActivity,
      coherence,
    };
  }

  private calculateDistance(
    pos1: { x: number; y: number; z?: number },
    pos2: { x: number; y: number; z?: number }
  ): number {
    const dx = pos1.x - pos2.x;
    const dy = pos1.y - pos2.y;
    const dz = (pos1.z || 0) - (pos2.z || 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private calculateCentroid(positions: Array<{ x: number; y: number; z?: number }>): { x: number; y: number; z?: number } {
    const sum = positions.reduce(
      (acc, pos) => ({
        x: acc.x + pos.x,
        y: acc.y + pos.y,
        z: acc.z + (pos.z || 0),
      }),
      { x: 0, y: 0, z: 0 }
    );

    return {
      x: sum.x / positions.length,
      y: sum.y / positions.length,
      z: positions.some(p => p.z !== undefined) ? sum.z / positions.length : undefined,
    };
  }

  private calculateSpatialCoherence(nodes: NeuralActivity[]): number {
    if (nodes.length < 2) return 1.0;

    const activations = nodes.map(n => n.activation);
    const mean = activations.reduce((sum, a) => sum + a, 0) / activations.length;
    const variance = activations.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / activations.length;
    
    return mean > 0 ? 1 / (1 + variance / (mean * mean)) : 0;
  }

  detectActivityWaves(
    currentActivities: NeuralActivity[],
    previousActivities: NeuralActivity[],
    timeInterval: number
  ): ActivityWave[] {
    const waves: ActivityWave[] = [];
    const threshold = 0.1; // Minimum activation change to detect wave
    const minWaveSize = 5; // Minimum number of nodes in a wave

    // Find nodes with significant activation changes
    const changedNodes = currentActivities.filter(current => {
      const previous = previousActivities.find(p => p.nodeId === current.nodeId);
      return previous && (current.activation - previous.activation) > threshold;
    });

    if (changedNodes.length < minWaveSize) return waves;

    // Group nodes by spatial proximity and temporal coherence
    const processed = new Set<string>();
    
    for (const node of changedNodes) {
      if (processed.has(node.nodeId)) continue;

      const waveNodes = this.findWaveNodes(node, changedNodes, processed);
      if (waveNodes.length >= minWaveSize) {
        const wave = this.createActivityWave(waveNodes, timeInterval);
        waves.push(wave);
      }
    }

    return waves;
  }

  private findWaveNodes(
    seed: NeuralActivity,
    candidates: NeuralActivity[],
    processed: Set<string>
  ): NeuralActivity[] {
    const waveNodes: NeuralActivity[] = [seed];
    const queue: NeuralActivity[] = [seed];
    processed.add(seed.nodeId);

    const waveRadius = this.resolution * 3;

    while (queue.length > 0) {
      const current = queue.shift()!;
      
      for (const candidate of candidates) {
        if (processed.has(candidate.nodeId)) continue;
        
        const distance = this.calculateDistance(current.position, candidate.position);
        if (distance <= waveRadius) {
          waveNodes.push(candidate);
          queue.push(candidate);
          processed.add(candidate.nodeId);
        }
      }
    }

    return waveNodes;
  }

  private createActivityWave(nodes: NeuralActivity[], timeInterval: number): ActivityWave {
    const centroid = this.calculateCentroid(nodes.map(n => n.position));
    const avgActivation = nodes.reduce((sum, n) => sum + n.activation, 0) / nodes.length;
    
    // Calculate propagation direction (simplified)
    const positions = nodes.map(n => n.position);
    const direction = this.calculatePropagationDirection(positions);
    
    // Estimate propagation speed (simplified - in practice this would need temporal tracking)
    const maxDistance = Math.max(...nodes.map(n => 
      this.calculateDistance(n.position, centroid)
    ));
    const propagationSpeed = maxDistance / (timeInterval / 1000); // pixels per second

    return {
      id: `wave_${Date.now()}_${nodes.length}`,
      origin: centroid,
      propagationSpeed,
      amplitude: avgActivation,
      wavelength: maxDistance * 2,
      direction,
      affectedNodes: nodes.map(n => n.nodeId),
      timestamp: Date.now(),
      duration: timeInterval,
    };
  }

  private calculatePropagationDirection(positions: Array<{ x: number; y: number; z?: number }>): { x: number; y: number; z?: number } {
    if (positions.length < 2) return { x: 0, y: 0, z: 0 };

    // Simplified direction calculation based on position spread
    const centroid = this.calculateCentroid(positions);
    const vectors = positions.map(pos => ({
      x: pos.x - centroid.x,
      y: pos.y - centroid.y,
      z: (pos.z || 0) - (centroid.z || 0),
    }));

    const avgVector = vectors.reduce(
      (acc, vec) => ({
        x: acc.x + vec.x,
        y: acc.y + vec.y,
        z: acc.z + vec.z,
      }),
      { x: 0, y: 0, z: 0 }
    );

    const length = Math.sqrt(avgVector.x ** 2 + avgVector.y ** 2 + avgVector.z ** 2);
    
    return length > 0 ? {
      x: avgVector.x / length,
      y: avgVector.y / length,
      z: avgVector.z / length,
    } : { x: 0, y: 0, z: 0 };
  }
}

// Spike Detection Engine
class SpikeDetectionEngine {
  private activityHistory: Map<string, Array<{ timestamp: number; activation: number }>> = new Map();
  private readonly historySize: number;
  private thresholds: Map<string, number> = new Map();

  constructor(historySize: number = 50) {
    this.historySize = historySize;
  }

  updateActivities(activities: NeuralActivity[]): SpikeEvent[] {
    const spikes: SpikeEvent[] = [];
    const timestamp = Date.now();

    activities.forEach(activity => {
      const history = this.activityHistory.get(activity.nodeId) || [];
      history.push({ timestamp, activation: activity.activation });

      // Keep only recent history
      if (history.length > this.historySize) {
        history.shift();
      }

      this.activityHistory.set(activity.nodeId, history);

      // Update threshold
      this.updateThreshold(activity.nodeId, history);

      // Detect spike
      const spike = this.detectSpike(activity, history);
      if (spike) {
        spikes.push(spike);
      }
    });

    return spikes;
  }

  private updateThreshold(nodeId: string, history: Array<{ timestamp: number; activation: number }>): void {
    if (history.length < 5) return;

    const activations = history.map(h => h.activation);
    const mean = activations.reduce((sum, a) => sum + a, 0) / activations.length;
    const stdDev = Math.sqrt(
      activations.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / activations.length
    );

    // Set threshold at mean + 2 * standard deviation
    this.thresholds.set(nodeId, mean + 2 * stdDev);
  }

  private detectSpike(
    activity: NeuralActivity,
    history: Array<{ timestamp: number; activation: number }>
  ): SpikeEvent | null {
    if (history.length < 3) return null;

    const threshold = this.thresholds.get(activity.nodeId) || 0.7;
    const current = history[history.length - 1];
    const previous = history[history.length - 2];

    // Simple spike detection: activation crosses threshold with sufficient derivative
    if (current.activation > threshold && 
        previous.activation <= threshold &&
        current.activation - previous.activation > 0.1) {
      
      // Calculate spike duration
      let duration = 0;
      for (let i = history.length - 1; i >= 0; i--) {
        if (history[i].activation > threshold) {
          duration = current.timestamp - history[i].timestamp;
        } else {
          break;
        }
      }

      return {
        timestamp: current.timestamp,
        nodeId: activity.nodeId,
        amplitude: current.activation,
        duration: Math.max(duration, 1), // At least 1ms
        layer: activity.layer,
        position: activity.position,
        presynapticNodes: [], // Would need connection data
        postsynapticNodes: [], // Would need connection data
      };
    }

    return null;
  }
}

// Main Hook Implementation
export const useNeuralActivity = (options: NeuralActivityOptions = {}) => {
  const {
    historySize = 200,
    layerFocusDepth = 5,
    connectionThreshold = 0.3,
    enableSpikeDetection = true,
    activityWindowMs = 5000,
    spatialResolution = 10,
    temporalSmoothing = 0.1,
    samplingRate = 60, // 60Hz for neural activity
  } = options;

  // Use base real-time data hook
  const {
    data: rawNeuralData,
    latest: latestNeural,
    isConnected,
    metrics: baseMetrics,
    aggregatedData,
  } = useRealTimeData<NeuralData>('neural', {
    bufferSize: historySize,
    samplingRate,
    aggregationWindow: activityWindowMs,
    enablePerformanceMonitoring: true,
  });

  // Analysis engines
  const [spatialEngine] = useState(() => new SpatialAnalysisEngine(spatialResolution));
  const [spikeEngine] = useState(() => new SpikeDetectionEngine(historySize));

  // State management
  const [spikeEvents, setSpikeEvents] = useState<SpikeEvent[]>([]);
  const [activityWaves, setActivityWaves] = useState<ActivityWave[]>([]);
  const [layerAnalyses, setLayerAnalyses] = useState<LayerAnalysis[]>([]);
  
  // Refs for tracking
  const previousActivitiesRef = useRef<NeuralActivity[]>([]);
  const layerHistoryRef = useRef<Map<string, Array<{ timestamp: number; activity: number }>>>(new Map());

  // Process neural activity data
  useEffect(() => {
    if (!latestNeural) return;

    const { activity: activities, layers, connections } = latestNeural;
    const timestamp = Date.now();

    // Update spatial analysis
    spatialEngine.updateNodePositions(activities);

    // Detect spikes
    if (enableSpikeDetection) {
      const newSpikes = spikeEngine.updateActivities(activities);
      setSpikeEvents(prev => {
        const updated = [...prev, ...newSpikes];
        return updated.slice(-historySize); // Keep only recent spikes
      });
    }

    // Detect activity waves
    const previousActivities = previousActivitiesRef.current;
    if (previousActivities.length > 0) {
      const waves = spatialEngine.detectActivityWaves(activities, previousActivities, 100); // Assume 100ms interval
      setActivityWaves(prev => {
        const updated = [...prev, ...waves];
        return updated.slice(-50); // Keep only recent waves
      });
    }

    // Update layer analyses
    const newLayerAnalyses = layers.map(layer => 
      analyzeLayer(layer, activities, rawNeuralData, connections, spatialEngine)
    );
    setLayerAnalyses(newLayerAnalyses);

    // Update layer history
    layers.forEach(layer => {
      const history = layerHistoryRef.current.get(layer.id) || [];
      history.push({ timestamp, activity: layer.averageActivation });
      
      if (history.length > historySize) {
        history.shift();
      }
      
      layerHistoryRef.current.set(layer.id, history);
    });

    // Update previous activities
    previousActivitiesRef.current = activities;
  }, [latestNeural, enableSpikeDetection, spatialEngine, spikeEngine, rawNeuralData, historySize]);

  // Calculate connectivity matrix
  const connectivityMatrix = useMemo((): ConnectivityMatrix => {
    if (!latestNeural || !latestNeural.connections.length) {
      return {
        layerPairs: [],
        overallConnectivity: 0,
        networkDensity: 0,
        smallWorldIndex: 0,
        modularity: 0,
      };
    }

    const { connections, layers } = latestNeural;
    const layerPairs: ConnectivityMatrix['layerPairs'] = [];

    // Group connections by layer pairs
    const connectionsByLayers = new Map<string, NeuralConnection[]>();

    for (const connection of connections) {
      // Find layers for nodes (simplified - in practice you'd have node-to-layer mapping)
      const fromLayer = layers[0]?.id || 'unknown'; // Simplified
      const toLayer = layers[1]?.id || 'unknown';   // Simplified
      const pairKey = `${fromLayer}-${toLayer}`;

      const pairConnections = connectionsByLayers.get(pairKey) || [];
      pairConnections.push(connection);
      connectionsByLayers.set(pairKey, pairConnections);
    }

    // Analyze each layer pair
    connectionsByLayers.forEach((connections, pairKey) => {
      const [fromLayer, toLayer] = pairKey.split('-');
      
      const activeConnections = connections.filter(c => c.active);
      const averageWeight = connections.reduce((sum, c) => sum + c.weight, 0) / connections.length;
      const activeRatio = activeConnections.length / connections.length;
      
      // Calculate synchrony (simplified)
      const synchrony = calculateConnectionSynchrony(connections);

      layerPairs.push({
        fromLayer,
        toLayer,
        connections: connections.map(c => ({
          strength: Math.abs(c.weight),
          active: c.active,
          weight: c.weight,
          delay: 0, // Would need temporal analysis
        })),
        averageWeight,
        activeRatio,
        synchrony,
      });
    });

    // Calculate overall metrics
    const totalConnections = connections.length;
    const activeConnections = connections.filter(c => c.active).length;
    const overallConnectivity = totalConnections > 0 ? activeConnections / totalConnections : 0;
    
    const maxPossibleConnections = layers.length * (layers.length - 1);
    const networkDensity = maxPossibleConnections > 0 ? totalConnections / maxPossibleConnections : 0;

    // Simplified metrics (in practice, these would need more sophisticated algorithms)
    const smallWorldIndex = calculateSmallWorldIndex(connections, layers);
    const modularity = calculateModularity(connections, layers);

    return {
      layerPairs,
      overallConnectivity,
      networkDensity,
      smallWorldIndex,
      modularity,
    };
  }, [latestNeural]);

  // Calculate neural metrics
  const neuralMetrics = useMemo((): NeuralMetrics => {
    if (!latestNeural || rawNeuralData.length === 0) {
      return {
        overallActivity: 0,
        layerActivation: {},
        connectionStrength: 0,
        networkEfficiency: 0,
        spikeFrequency: 0,
        spatialCoherence: 0,
        temporalStability: 0,
        plasticity: 0,
        energyConsumption: 0,
      };
    }

    const { activity, layers, connections } = latestNeural;
    
    const overallActivity = activity.reduce((sum, a) => sum + a.activation, 0) / activity.length || 0;
    
    const layerActivation = layers.reduce((acc, layer) => {
      acc[layer.id] = layer.averageActivation;
      return acc;
    }, {} as Record<string, number>);

    const connectionStrength = connections.reduce((sum, c) => sum + Math.abs(c.weight), 0) / connections.length || 0;
    
    const networkEfficiency = connectivityMatrix.overallConnectivity * connectivityMatrix.networkDensity;
    
    const recentSpikes = spikeEvents.filter(spike => Date.now() - spike.timestamp <= activityWindowMs);
    const spikeFrequency = (recentSpikes.length * 1000) / activityWindowMs;
    
    const spatialCoherence = calculateSpatialCoherence(activity);
    const temporalStability = calculateTemporalStability(rawNeuralData);
    
    // Simplified plasticity measure based on connection weight changes
    const plasticity = calculatePlasticity(rawNeuralData);
    
    // Energy consumption estimate based on activity and connections
    const energyConsumption = overallActivity * connectionStrength;

    return {
      overallActivity,
      layerActivation,
      connectionStrength,
      networkEfficiency,
      spikeFrequency,
      spatialCoherence,
      temporalStability,
      plasticity,
      energyConsumption,
    };
  }, [latestNeural, rawNeuralData, connectivityMatrix, spikeEvents, activityWindowMs]);

  return {
    // Core data
    activities: latestNeural?.activity || [],
    layers: layerAnalyses,
    connections: latestNeural?.connections || [],
    spikeEvents,
    activityWaves,
    
    // Analysis results
    connectivityMatrix,
    metrics: neuralMetrics,
    isConnected,
    
    // Aggregated and historical data
    historicalData: rawNeuralData,
    aggregatedData,
    
    // Control methods
    clearHistory: useCallback(() => {
      setSpikeEvents([]);
      setActivityWaves([]);
      layerHistoryRef.current.clear();
    }, []),
    
    findSpatialClusters: useCallback((minClusterSize: number = 3) => {
      if (!latestNeural) return [];
      return spatialEngine.findSpatialClusters(latestNeural.activity, minClusterSize);
    }, [latestNeural, spatialEngine]),
    
    getLayerTrend: useCallback((layerId: string) => {
      const history = layerHistoryRef.current.get(layerId);
      if (!history || history.length < 3) return 'stable';
      
      const values = history.slice(-10).map(h => h.activity); // Last 10 samples
      return calculateTrend(values);
    }, []),
  };
};

// Helper functions
function analyzeLayer(
  layer: NeuralLayer,
  activities: NeuralActivity[],
  historicalData: NeuralData[],
  connections: NeuralConnection[],
  spatialEngine: SpatialAnalysisEngine
): LayerAnalysis {
  const layerActivities = activities.filter(a => a.layer === parseInt(layer.id) || a.nodeId.startsWith(layer.id));
  
  const activeNodes = layerActivities.filter(a => a.activation > 0.1).length;
  const utilization = layer.nodeCount > 0 ? activeNodes / layer.nodeCount : 0;
  
  // Calculate spike rate (simplified)
  const spikeRate = layerActivities.filter(a => a.activation > 0.7).length / layer.nodeCount;
  
  // Calculate synchrony
  const synchrony = layerActivities.length > 1 ? calculateSynchrony(layerActivities) : 0;
  
  // Find spatial clusters
  const spatialClusters = spatialEngine.findSpatialClusters(layerActivities, 2);
  
  // Determine temporal pattern
  const temporalPattern = determineTemporalPattern(layer.id, historicalData);
  
  // Calculate trend
  const recentData = historicalData.slice(-10);
  const layerActivations = recentData.map(d => 
    d.layers.find(l => l.id === layer.id)?.averageActivation || 0
  );
  const activityTrend = calculateTrend(layerActivations);

  return {
    id: layer.id,
    name: layer.name,
    currentActivity: layer.averageActivation,
    averageActivity: layerActivations.reduce((sum, a) => sum + a, 0) / layerActivations.length || 0,
    peakActivity: Math.max(...layerActivations),
    activityTrend,
    nodeCount: layer.nodeCount,
    activeNodes,
    utilization,
    spikeRate,
    synchrony,
    spatialClusters,
    temporalPattern,
  };
}

function calculateConnectionSynchrony(connections: NeuralConnection[]): number {
  // Simplified synchrony calculation based on weight correlation
  if (connections.length < 2) return 1.0;
  
  const weights = connections.map(c => c.weight);
  const mean = weights.reduce((sum, w) => sum + w, 0) / weights.length;
  const variance = weights.reduce((sum, w) => sum + Math.pow(w - mean, 2), 0) / weights.length;
  
  return mean !== 0 ? 1 / (1 + variance / (mean * mean)) : 0;
}

function calculateSmallWorldIndex(connections: NeuralConnection[], layers: NeuralLayer[]): number {
  // Simplified small world calculation
  // In practice, this would require graph analysis algorithms
  const clustering = calculateClusteringCoefficient(connections);
  const pathLength = calculateAveragePathLength(connections, layers);
  
  return clustering / Math.max(pathLength, 0.001);
}

function calculateModularity(connections: NeuralConnection[], layers: NeuralLayer[]): number {
  // Simplified modularity calculation
  // In practice, this would use community detection algorithms
  const totalConnections = connections.length;
  const interLayerConnections = connections.filter(c => 
    // Simplified layer detection - in practice you'd have proper node-layer mapping
    c.from.split('_')[0] !== c.to.split('_')[0]
  ).length;
  
  return totalConnections > 0 ? 1 - (interLayerConnections / totalConnections) : 0;
}

function calculateClusteringCoefficient(connections: NeuralConnection[]): number {
  // Simplified clustering coefficient
  return connections.filter(c => c.active).length / Math.max(connections.length, 1);
}

function calculateAveragePathLength(connections: NeuralConnection[], layers: NeuralLayer[]): number {
  // Simplified path length calculation
  return layers.length > 1 ? layers.length / 2 : 1;
}

function calculateSpatialCoherence(activities: NeuralActivity[]): number {
  if (activities.length < 2) return 1.0;
  
  // Calculate variance in spatial distribution weighted by activation
  const totalActivation = activities.reduce((sum, a) => sum + a.activation, 0);
  if (totalActivation === 0) return 0;
  
  const weightedCentroid = activities.reduce(
    (acc, a) => ({
      x: acc.x + a.position.x * a.activation,
      y: acc.y + a.position.y * a.activation,
    }),
    { x: 0, y: 0 }
  );
  
  weightedCentroid.x /= totalActivation;
  weightedCentroid.y /= totalActivation;
  
  const variance = activities.reduce((sum, a) => {
    const dx = a.position.x - weightedCentroid.x;
    const dy = a.position.y - weightedCentroid.y;
    return sum + a.activation * (dx * dx + dy * dy);
  }, 0) / totalActivation;
  
  return 1 / (1 + variance / 100); // Normalize by arbitrary scale
}

function calculateTemporalStability(historicalData: NeuralData[]): number {
  if (historicalData.length < 3) return 1.0;
  
  const activities = historicalData.map(d => d.overallActivity);
  const mean = activities.reduce((sum, a) => sum + a, 0) / activities.length;
  const variance = activities.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / activities.length;
  
  return mean > 0 ? 1 / (1 + variance / (mean * mean)) : 0;
}

function calculatePlasticity(historicalData: NeuralData[]): number {
  if (historicalData.length < 2) return 0;
  
  const recent = historicalData.slice(-5);
  let weightChanges = 0;
  let totalConnections = 0;
  
  for (let i = 1; i < recent.length; i++) {
    const prev = recent[i - 1].connections;
    const curr = recent[i].connections;
    
    totalConnections += curr.length;
    
    for (const connection of curr) {
      const prevConnection = prev.find(c => c.from === connection.from && c.to === connection.to);
      if (prevConnection) {
        weightChanges += Math.abs(connection.weight - prevConnection.weight);
      }
    }
  }
  
  return totalConnections > 0 ? weightChanges / totalConnections : 0;
}

function calculateSynchrony(activities: NeuralActivity[]): number {
  if (activities.length < 2) return 1.0;
  
  const activations = activities.map(a => a.activation);
  const mean = activations.reduce((sum, a) => sum + a, 0) / activations.length;
  const variance = activations.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / activations.length;
  
  return mean > 0 ? 1 / (1 + variance / (mean * mean)) : 0;
}

function determineTemporalPattern(layerId: string, historicalData: NeuralData[]): 'burst' | 'tonic' | 'oscillatory' | 'sparse' {
  if (historicalData.length < 10) return 'sparse';
  
  const layerActivations = historicalData
    .map(d => d.layers.find(l => l.id === layerId)?.averageActivation || 0)
    .slice(-20); // Last 20 samples
  
  const mean = layerActivations.reduce((sum, a) => sum + a, 0) / layerActivations.length;
  const variance = layerActivations.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / layerActivations.length;
  const cv = mean > 0 ? Math.sqrt(variance) / mean : 0;
  
  // Simple pattern classification
  if (mean < 0.1) return 'sparse';
  if (cv > 1.5) return 'burst';
  if (cv < 0.3) return 'tonic';
  
  // Check for oscillations (simplified)
  const peaks = layerActivations.filter((a, i) => 
    i > 0 && i < layerActivations.length - 1 && 
    a > layerActivations[i - 1] && a > layerActivations[i + 1]
  ).length;
  
  return peaks > 3 ? 'oscillatory' : 'tonic';
}

function calculateTrend(values: number[]): 'rising' | 'falling' | 'stable' {
  if (values.length < 2) return 'stable';
  
  const first = values.slice(0, Math.ceil(values.length / 2));
  const second = values.slice(Math.floor(values.length / 2));
  
  const firstAvg = first.reduce((sum, v) => sum + v, 0) / first.length;
  const secondAvg = second.reduce((sum, v) => sum + v, 0) / second.length;
  
  const threshold = 0.05; // 5% change threshold
  const change = (secondAvg - firstAvg) / Math.abs(firstAvg || 1);
  
  if (change > threshold) return 'rising';
  if (change < -threshold) return 'falling';
  return 'stable';
}

// Export types
export type {
  NeuralActivityOptions,
  SpikeEvent,
  LayerAnalysis,
  SpatialCluster,
  ConnectivityMatrix,
  ActivityWave,
  NeuralMetrics,
};