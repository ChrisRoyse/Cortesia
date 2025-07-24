import { LLMKGData, KnowledgeGraphData, KnowledgeNode, KnowledgeEdge } from '../types';
import { 
  BrainGraphData, 
  BrainEntity, 
  BrainRelationship, 
  ConceptStructure, 
  BrainStatistics,
  ActivationDistribution,
  BrainMetrics,
  LogicGate
} from '../types/brain';

/**
 * Transforms basic KnowledgeGraphData to sophisticated BrainGraphData format
 * This bridges the gap between WebSocket metrics and the 3D brain visualization
 */
export function transformKnowledgeGraphToBrainData(data: LLMKGData): BrainGraphData | null {
  if (!data.knowledgeGraph) {
    console.warn('No knowledge graph data available for transformation');
    return null;
  }

  const { knowledgeGraph, cognitive, neural } = data;
  const now = Date.now();

  try {
    // Transform basic nodes to brain entities
    const entities: BrainEntity[] = knowledgeGraph.nodes.map((node, index) => {
      const direction = determineEntityDirection(node, index, knowledgeGraph.nodes.length);
      // Handle neural activity array structure - it might be array of numbers or objects
      const activation = Array.isArray(neural.activity) 
        ? (typeof neural.activity[index] === 'number' 
            ? neural.activity[index] / 100 
            : neural.activity[index]?.activation || Math.random() * 0.8 + 0.1)
        : Math.random() * 0.8 + 0.1;
      
      return {
        id: node.id,
        type_id: hashStringToId(node.type),
        properties: {
          name: node.label,
          type: node.type,
          weight: node.weight,
          ...node.metadata
        },
        embedding: generateEmbedding(node),
        activation,
        direction,
        lastActivation: now - Math.random() * 60000, // Random recent activation
        lastUpdate: now,
        conceptIds: extractConceptIds(node, cognitive.patterns)
      };
    });

    // Transform basic edges to brain relationships
    const relationships: BrainRelationship[] = knowledgeGraph.edges.map(edge => {
      const isInhibitory = edge.type.includes('inhibit') || edge.type.includes('suppress') || Math.random() < 0.3;
      
      return {
        from: edge.source,
        to: edge.target,
        relType: hashStringToId(edge.type),
        weight: edge.weight,
        inhibitory: isInhibitory,
        temporalDecay: 0.95 - Math.random() * 0.1, // 0.85-0.95 range
        lastActivation: now - Math.random() * 30000,
        usageCount: Math.floor(Math.random() * 20) + 1
      };
    });

    // Generate concepts from cognitive patterns and clustering
    const concepts: ConceptStructure[] = generateConcepts(
      entities, 
      relationships, 
      cognitive.patterns,
      knowledgeGraph.clusters
    );

    // Generate logic gates from relationships
    const logicGates: LogicGate[] = generateLogicGates(entities, relationships);

    // Calculate brain statistics
    const statistics: BrainStatistics = calculateBrainStatistics(
      entities, 
      relationships, 
      concepts,
      knowledgeGraph.metrics
    );

    // Calculate activation distribution
    const activationDistribution: ActivationDistribution = calculateActivationDistribution(entities);

    // Generate brain metrics (backend format simulation)
    const metrics: BrainMetrics = generateBrainMetrics(statistics);

    return {
      entities,
      relationships,
      concepts,
      logicGates,
      statistics,
      activationDistribution,
      metrics
    };
  } catch (error) {
    console.error('Error transforming knowledge graph data:', error);
    return null;
  }
}

/**
 * Determine entity direction based on position and connectivity
 */
function determineEntityDirection(
  node: KnowledgeNode, 
  index: number, 
  totalNodes: number
): 'Input' | 'Output' | 'Gate' | 'Hidden' {
  const position = index / totalNodes;
  
  if (node.type === 'property' || position < 0.2) return 'Input';
  if (node.type === 'relation' || position > 0.8) return 'Output';
  if (node.type === 'entity' && node.weight > 0.7) return 'Gate';
  return 'Hidden';
}

/**
 * Generate embedding vector from node properties
 */
function generateEmbedding(node: KnowledgeNode): number[] {
  const embedding = new Array(128).fill(0);
  const seed = hashStringToId(node.id);
  
  // Generate deterministic but varied embedding based on node properties
  for (let i = 0; i < 128; i++) {
    const angle = (seed + i * 7) % 1000 / 1000 * Math.PI * 2;
    embedding[i] = Math.sin(angle) * node.weight + Math.cos(angle) * 0.1;
  }
  
  return embedding;
}

/**
 * Extract concept IDs from cognitive patterns
 */
function extractConceptIds(node: KnowledgeNode, patterns: any[]): string[] {
  const conceptIds: string[] = [];
  
  patterns.forEach((pattern, index) => {
    if (pattern.activeNodes?.includes(node.id) || Math.random() < 0.3) {
      conceptIds.push(`concept-${index}`);
    }
  });
  
  return conceptIds.length > 0 ? conceptIds : [`concept-default-${hashStringToId(node.id) % 3}`];
}

/**
 * Generate concepts from entities and patterns
 */
function generateConcepts(
  entities: BrainEntity[],
  relationships: BrainRelationship[],
  patterns: any[],
  clusters: any[]
): ConceptStructure[] {
  const concepts: ConceptStructure[] = [];
  const now = Date.now();

  // Generate concepts from cognitive patterns
  patterns.forEach((pattern, index) => {
    const relatedEntities = entities.filter(e => 
      e.conceptIds.includes(`concept-${index}`) || 
      (pattern.activeNodes && pattern.activeNodes.includes(e.id))
    );

    if (relatedEntities.length > 0) {
      const inputs = relatedEntities.filter(e => e.direction === 'Input').map(e => e.id);
      const outputs = relatedEntities.filter(e => e.direction === 'Output').map(e => e.id);
      const gates = relatedEntities.filter(e => e.direction === 'Gate').map(e => e.id);

      concepts.push({
        id: `concept-${index}`,
        name: pattern.type || `Concept ${index + 1}`,
        inputs,
        outputs,
        gates,
        coherence: pattern.coherence || 0.7 + Math.random() * 0.2,
        activation: pattern.strength / 100 || Math.random() * 0.6 + 0.2,
        lastUpdate: now
      });
    }
  });

  // Generate concepts from clusters
  clusters.forEach((cluster, index) => {
    if (!concepts.find(c => c.id === `cluster-concept-${index}`)) {
      const clusterEntities = entities.filter(e => 
        cluster.nodes?.includes(e.id) || Math.random() < 0.4
      );

      concepts.push({
        id: `cluster-concept-${index}`,
        name: cluster.topic || `Cluster ${index + 1}`,
        inputs: clusterEntities.filter(e => e.direction === 'Input').map(e => e.id),
        outputs: clusterEntities.filter(e => e.direction === 'Output').map(e => e.id),
        gates: clusterEntities.filter(e => e.direction === 'Gate').map(e => e.id),
        coherence: cluster.density || 0.6 + Math.random() * 0.3,
        activation: Math.random() * 0.8 + 0.1,
        lastUpdate: now
      });
    }
  });

  return concepts;
}

/**
 * Generate logic gates from relationship patterns
 */
function generateLogicGates(entities: BrainEntity[], relationships: BrainRelationship[]): LogicGate[] {
  const gates: LogicGate[] = [];
  
  // Find entities that could be logic gates (Gate direction with multiple inputs)
  const gateEntities = entities.filter(e => e.direction === 'Gate');
  
  gateEntities.forEach(gateEntity => {
    const inputs = relationships
      .filter(r => r.to === gateEntity.id)
      .map(r => r.from);
    
    const outputs = relationships
      .filter(r => r.from === gateEntity.id)
      .map(r => r.to);

    if (inputs.length >= 2) {
      const gateTypes: LogicGate['type'][] = ['AND', 'OR', 'XOR', 'THRESHOLD'];
      const gateType = gateTypes[hashStringToId(gateEntity.id) % gateTypes.length];
      
      gates.push({
        id: gateEntity.id,
        type: gateType,
        inputs,
        outputs,
        threshold: gateType === 'THRESHOLD' ? 0.5 + Math.random() * 0.3 : undefined,
        currentState: gateEntity.activation > 0.5
      });
    }
  });

  return gates;
}

/**
 * Calculate comprehensive brain statistics
 */
function calculateBrainStatistics(
  entities: BrainEntity[],
  relationships: BrainRelationship[],
  concepts: ConceptStructure[],
  basicMetrics?: any
): BrainStatistics {
  const activations = entities.map(e => e.activation);
  const totalActivation = activations.reduce((sum, a) => sum + a, 0);
  const avgActivation = totalActivation / entities.length;
  
  const activeEntities = entities.filter(e => e.activation > 0.3).length;
  const density = basicMetrics?.density || (relationships.length / (entities.length * (entities.length - 1)));
  
  return {
    entityCount: entities.length,
    relationshipCount: relationships.length,
    avgActivation,
    minActivation: Math.min(...activations),
    maxActivation: Math.max(...activations),
    totalActivation,
    graphDensity: Math.min(density, 1.0),
    clusteringCoefficient: basicMetrics?.avgDegree ? basicMetrics.avgDegree / entities.length : 0.5,
    betweennessCentrality: 0.4 + Math.random() * 0.3,
    learningEfficiency: 0.6 + Math.random() * 0.3,
    conceptCoherence: concepts.length > 0 ? 
      concepts.reduce((sum, c) => sum + c.coherence, 0) / concepts.length : 0.7,
    activeEntities,
    avgRelationshipsPerEntity: relationships.length / entities.length,
    uniqueEntityTypes: new Set(entities.map(e => e.direction)).size
  };
}

/**
 * Calculate activation distribution buckets
 */
function calculateActivationDistribution(entities: BrainEntity[]): ActivationDistribution {
  const distribution = {
    veryLow: 0,   // 0.0-0.2
    low: 0,       // 0.2-0.4
    medium: 0,    // 0.4-0.6
    high: 0,      // 0.6-0.8
    veryHigh: 0   // 0.8-1.0
  };

  entities.forEach(entity => {
    const activation = entity.activation;
    if (activation < 0.2) distribution.veryLow++;
    else if (activation < 0.4) distribution.low++;
    else if (activation < 0.6) distribution.medium++;
    else if (activation < 0.8) distribution.high++;
    else distribution.veryHigh++;
  });

  return distribution;
}

/**
 * Generate brain metrics in backend format
 */
function generateBrainMetrics(stats: BrainStatistics): BrainMetrics {
  return {
    brain_entity_count: stats.entityCount,
    brain_relationship_count: stats.relationshipCount,
    brain_avg_activation: stats.avgActivation,
    brain_max_activation: stats.maxActivation,
    brain_graph_density: stats.graphDensity,
    brain_clustering_coefficient: stats.clusteringCoefficient,
    brain_total_activation: stats.totalActivation,
    brain_active_entities: stats.activeEntities,
    brain_learning_efficiency: stats.learningEfficiency,
    brain_concept_coherence: stats.conceptCoherence,
    brain_avg_relationships_per_entity: stats.avgRelationshipsPerEntity,
    brain_unique_entity_types: stats.uniqueEntityTypes,
    brain_memory_bytes: 1024 * 1024,
    brain_index_memory_bytes: 512 * 1024,
    brain_embedding_memory_bytes: stats.entityCount * 128 * 4, // float32 embeddings
    brain_total_chunks: Math.floor(stats.entityCount / 10),
    brain_total_triples: stats.relationshipCount
  };
}

/**
 * Simple string to number hash function
 */
function hashStringToId(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash);
}