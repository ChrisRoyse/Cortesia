export interface CognitivePattern {
  id: string;
  type: PatternType;
  name: string;
  activation: number;
  confidence: number;
  timestamp: number;
  connections: PatternConnection[];
  metadata: PatternMetadata;
}

export type PatternType = 
  | 'convergent'
  | 'divergent'
  | 'lateral'
  | 'systems'
  | 'critical'
  | 'abstract'
  | 'adaptive'
  | 'chain_of_thought'
  | 'tree_of_thoughts';

export interface PatternConnection {
  sourceId: string;
  targetId: string;
  strength: number;
  type: 'excitatory' | 'inhibitory';
}

export interface PatternMetadata {
  complexity: number;
  resourceUsage: {
    cpu: number;
    memory: number;
    duration: number;
  };
  parameters: Record<string, any>;
  tags: string[];
}

export interface PatternActivation {
  patternId: string;
  timestamp: number;
  activationLevel: number;
  propagation: PropagationStep[];
  outcome: 'success' | 'failure' | 'partial';
}

export interface PropagationStep {
  nodeId: string;
  activation: number;
  timestamp: number;
  depth: number;
}

export interface InhibitionExcitationBalance {
  timestamp: number;
  excitation: {
    total: number;
    byRegion: Record<string, number>;
    patterns: string[];
  };
  inhibition: {
    total: number;
    byRegion: Record<string, number>;
    patterns: string[];
  };
  balance: number; // -1 (full inhibition) to 1 (full excitation)
  optimalRange: [number, number];
}

export interface TemporalPattern {
  id: string;
  sequence: TemporalEvent[];
  frequency: number;
  duration: number;
  predictability: number;
  nextPredicted?: TemporalEvent;
}

export interface TemporalEvent {
  patternId: string;
  timestamp: number;
  activation: number;
  context: string[];
}

export interface CognitiveMetrics {
  totalPatterns: number;
  activePatterns: number;
  averageActivation: number;
  patternDistribution: Record<PatternType, number>;
  performanceMetrics: {
    successRate: number;
    averageLatency: number;
    resourceEfficiency: number;
  };
}

export interface PatternCluster {
  id: string;
  patterns: string[];
  centroid: {
    x: number;
    y: number;
    z: number;
  };
  radius: number;
  density: number;
  dominantType: PatternType;
}

export interface ActivationHeatmap {
  width: number;
  height: number;
  data: number[][];
  regions: HeatmapRegion[];
}

export interface HeatmapRegion {
  name: string;
  bounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  averageActivation: number;
}