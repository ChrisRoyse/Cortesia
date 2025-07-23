// Brain-Enhanced Knowledge Graph Types matching Rust structures

export interface BrainEntity {
  id: string;
  type_id: number;
  properties: Record<string, any>;
  embedding: number[];
  activation: number; // 0.0-1.0
  direction: 'Input' | 'Output' | 'Gate' | 'Hidden';
  lastActivation: number; // timestamp
  lastUpdate: number; // timestamp
  conceptIds: string[];
}

export interface BrainRelationship {
  from: string;
  to: string;
  relType: number;
  weight: number;
  inhibitory: boolean;
  temporalDecay: number;
  lastActivation: number;
  usageCount: number;
}

export interface LogicGate {
  id: string;
  type: 'AND' | 'OR' | 'NOT' | 'XOR' | 'NAND' | 'NOR' | 'XNOR' | 'IDENTITY' | 'THRESHOLD';
  inputs: string[];
  outputs: string[];
  threshold?: number;
  currentState: boolean;
}

export interface BrainStatistics {
  entityCount: number;
  relationshipCount: number;
  avgActivation: number;
  minActivation: number;
  maxActivation: number;
  totalActivation: number;
  graphDensity: number;
  clusteringCoefficient: number;
  betweennessCentrality: number;
  learningEfficiency: number;
  conceptCoherence: number;
  activeEntities: number;
  avgRelationshipsPerEntity: number;
  uniqueEntityTypes: number;
}

export interface ActivationDistribution {
  veryLow: number;  // 0.0-0.2
  low: number;      // 0.2-0.4
  medium: number;   // 0.4-0.6
  high: number;     // 0.6-0.8
  veryHigh: number; // 0.8-1.0
}

export interface ConceptStructure {
  id: string;
  name: string;
  inputs: string[];
  outputs: string[];
  gates: string[];
  coherence: number;
  activation: number;
  lastUpdate: number;
}

export interface ActivationPropagationResult {
  activatedEntities: string[];
  propagationDepth: number;
  totalActivation: number;
  averageActivation: number;
  timeElapsed: number;
}

export interface BrainMetrics {
  brain_entity_count: number;
  brain_relationship_count: number;
  brain_avg_activation: number;
  brain_max_activation: number;
  brain_graph_density: number;
  brain_clustering_coefficient: number;
  brain_total_activation: number;
  brain_active_entities: number;
  brain_learning_efficiency: number;
  brain_concept_coherence: number;
  brain_avg_relationships_per_entity: number;
  brain_unique_entity_types: number;
  brain_memory_bytes: number;
  brain_index_memory_bytes: number;
  brain_embedding_memory_bytes: number;
  brain_total_chunks: number;
  brain_total_triples: number;
}

export interface BrainGraphData {
  entities: BrainEntity[];
  relationships: BrainRelationship[];
  concepts: ConceptStructure[];
  logicGates: LogicGate[];
  statistics: BrainStatistics;
  activationDistribution: ActivationDistribution;
  metrics: BrainMetrics;
}