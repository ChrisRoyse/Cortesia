/**
 * @fileoverview Knowledge Graph Data Collector for LLMKG Visualization
 * 
 * This module implements specialized data collection for LLMKG's knowledge graph operations.
 * It monitors entity operations, graph topology changes, triple relationships, and
 * semantic querying patterns with high-frequency data processing capabilities.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

import { BaseCollector, CollectedData, CollectorConfig, CollectionMetadata } from './base.js';
import { MCPClient } from '../mcp/client.js';
import { LLMKGTools } from '../mcp/types.js';

/**
 * Knowledge graph specific metrics
 */
export interface EntityMetrics {
  /** Number of entities in the graph */
  entityCount: number;
  /** Number of entity types */
  entityTypes: number;
  /** Average entity connections */
  avgConnections: number;
  /** Entity creation rate (per second) */
  creationRate: number;
  /** Entity deletion rate (per second) */
  deletionRate: number;
  /** Most connected entities */
  topEntities: EntityInfo[];
  /** Entity type distribution */
  typeDistribution: Record<string, number>;
}

/**
 * Graph topology metrics
 */
export interface TopologyData {
  /** Total number of nodes */
  nodeCount: number;
  /** Total number of edges */
  edgeCount: number;
  /** Graph density (edges / possible edges) */
  density: number;
  /** Number of connected components */
  connectedComponents: number;
  /** Average clustering coefficient */
  clusteringCoefficient: number;
  /** Graph diameter (longest shortest path) */
  diameter: number;
  /** Average path length */
  avgPathLength: number;
  /** Degree distribution */
  degreeDistribution: DegreeStats;
}

/**
 * Triple relationship metrics
 */
export interface TripleMetrics {
  /** Total number of triples */
  tripleCount: number;
  /** Number of unique predicates */
  predicateCount: number;
  /** Most frequent predicates */
  topPredicates: PredicateInfo[];
  /** Triple creation rate */
  creationRate: number;
  /** Validation success rate */
  validationRate: number;
  /** Subject-Predicate-Object distributions */
  distributions: {
    subjects: Record<string, number>;
    predicates: Record<string, number>;
    objects: Record<string, number>;
  };
}

/**
 * Query performance metrics
 */
export interface QueryMetrics {
  /** Total queries processed */
  queryCount: number;
  /** Average query execution time */
  avgExecutionTime: number;
  /** Query success rate */
  successRate: number;
  /** Most frequent query patterns */
  queryPatterns: QueryPattern[];
  /** Query complexity distribution */
  complexityDistribution: Record<string, number>;
  /** Cache hit rate */
  cacheHitRate: number;
  /** Query throughput (queries/second) */
  throughput: number;
}

/**
 * Supporting interfaces
 */
export interface EntityInfo {
  id: string;
  type: string;
  connectionCount: number;
  weight: number;
}

export interface DegreeStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  stdDev: number;
  histogram: Record<number, number>;
}

export interface PredicateInfo {
  predicate: string;
  count: number;
  percentage: number;
}

export interface QueryPattern {
  pattern: string;
  count: number;
  avgExecutionTime: number;
  successRate: number;
}

/**
 * Knowledge graph collector configuration
 */
export interface KnowledgeGraphCollectorConfig extends CollectorConfig {
  /** Enable entity operation monitoring */
  monitorEntities: boolean;
  /** Enable topology analysis */
  monitorTopology: boolean;
  /** Enable triple relationship tracking */
  monitorTriples: boolean;
  /** Enable query performance monitoring */
  monitorQueries: boolean;
  /** Topology analysis interval (ms) */
  topologyInterval: number;
  /** Maximum entities to track in detail */
  maxTrackedEntities: number;
  /** Query pattern cache size */
  queryPatternCacheSize: number;
}

/**
 * Specialized collector for LLMKG knowledge graph data
 */
export class KnowledgeGraphCollector extends BaseCollector {
  private entityCache = new Map<string, EntityInfo>();
  private tripleCache = new Map<string, number>();
  private queryPatternCache = new Map<string, QueryPattern>();
  private topologyCache: TopologyData | null = null;
  private lastTopologyUpdate = 0;
  private queryExecutionTimes: number[] = [];
  private tripleCreationTimes: number[] = [];
  
  private static readonly DEFAULT_KG_CONFIG: KnowledgeGraphCollectorConfig = {
    ...BaseCollector['DEFAULT_CONFIG'],
    name: 'knowledge-graph-collector',
    collectionInterval: 50, // 20 Hz for high-frequency graph operations
    monitorEntities: true,
    monitorTopology: true,
    monitorTriples: true,
    monitorQueries: true,
    topologyInterval: 10000, // 10 seconds
    maxTrackedEntities: 10000,
    queryPatternCacheSize: 1000
  };

  constructor(mcpClient: MCPClient, config: Partial<KnowledgeGraphCollectorConfig> = {}) {
    const mergedConfig = { ...KnowledgeGraphCollector.DEFAULT_KG_CONFIG, ...config };
    super(mcpClient, mergedConfig);
  }

  /**
   * Initializes the knowledge graph collector
   */
  async initialize(): Promise<void> {
    console.log(`Initializing Knowledge Graph Collector: ${this.config.name}`);
    
    try {
      // Setup MCP event handlers for graph operations
      this.setupGraphEventHandlers();
      
      // Initial graph state collection
      await this.collectInitialState();
      
      // Setup periodic topology analysis
      setInterval(async () => {
        await this.performTopologyAnalysis();
      }, (this.config as KnowledgeGraphCollectorConfig).topologyInterval);
      
      console.log(`Knowledge Graph Collector initialized successfully`);
    } catch (error) {
      console.error(`Failed to initialize Knowledge Graph Collector:`, error);
      throw error;
    }
  }

  /**
   * Cleans up resources
   */
  async cleanup(): Promise<void> {
    console.log(`Cleaning up Knowledge Graph Collector: ${this.config.name}`);
    
    this.entityCache.clear();
    this.tripleCache.clear();
    this.queryPatternCache.clear();
    this.queryExecutionTimes = [];
    this.tripleCreationTimes = [];
    this.topologyCache = null;
  }

  /**
   * Main collection method
   */
  async collect(): Promise<CollectedData[]> {
    const collectedData: CollectedData[] = [];
    const config = this.config as KnowledgeGraphCollectorConfig;

    try {
      // Collect entity metrics
      if (config.monitorEntities) {
        const entityMetrics = await this.collectEntityOperations();
        if (entityMetrics) {
          collectedData.push(this.createCollectedData('entity_metrics', entityMetrics, 'collectEntityOperations'));
        }
      }

      // Collect graph topology
      if (config.monitorTopology && this.shouldUpdateTopology()) {
        const topologyData = await this.collectGraphTopology();
        if (topologyData) {
          collectedData.push(this.createCollectedData('topology_data', topologyData, 'collectGraphTopology'));
        }
      }

      // Collect triple metrics
      if (config.monitorTriples) {
        const tripleMetrics = await this.collectTripleMetrics();
        if (tripleMetrics) {
          collectedData.push(this.createCollectedData('triple_metrics', tripleMetrics, 'collectTripleMetrics'));
        }
      }

      // Collect query metrics
      if (config.monitorQueries) {
        const queryMetrics = await this.collectQueryMetrics();
        if (queryMetrics) {
          collectedData.push(this.createCollectedData('query_metrics', queryMetrics, 'collectQueryMetrics'));
        }
      }

    } catch (error) {
      console.error(`Error in knowledge graph collection:`, error);
      this.emit('collection:error', error);
    }

    return collectedData;
  }

  /**
   * Collects entity operation metrics
   */
  async collectEntityOperations(): Promise<EntityMetrics | null> {
    try {
      // Query current entity state via MCP
      const entityData = await this.mcpClient.llmkg.knowledgeGraphQuery({
        query: 'SELECT ?entity ?type (COUNT(?connection) as ?connections) WHERE { ?entity rdf:type ?type . ?entity ?pred ?connection }',
        limit: (this.config as KnowledgeGraphCollectorConfig).maxTrackedEntities,
        includeWeights: true,
        entityTypes: ['concept', 'relation', 'attribute']
      });

      if (!entityData?.entities) {
        return null;
      }

      // Process entity data
      const entities = entityData.entities;
      const entityTypes = new Set<string>();
      const typeDistribution: Record<string, number> = {};
      let totalConnections = 0;
      const topEntities: EntityInfo[] = [];

      for (const entity of entities) {
        const entityInfo: EntityInfo = {
          id: entity.id,
          type: entity.type,
          connectionCount: entity.connections || 0,
          weight: entity.weight || 1
        };

        this.entityCache.set(entity.id, entityInfo);
        entityTypes.add(entity.type);
        typeDistribution[entity.type] = (typeDistribution[entity.type] || 0) + 1;
        totalConnections += entityInfo.connectionCount;

        if (topEntities.length < 10) {
          topEntities.push(entityInfo);
        } else {
          topEntities.sort((a, b) => b.connectionCount - a.connectionCount);
          if (entityInfo.connectionCount > topEntities[9].connectionCount) {
            topEntities[9] = entityInfo;
          }
        }
      }

      // Calculate rates
      const now = Date.now();
      const creationRate = this.calculateCreationRate('entity', now);
      const deletionRate = this.calculateDeletionRate('entity', now);

      return {
        entityCount: entities.length,
        entityTypes: entityTypes.size,
        avgConnections: entities.length > 0 ? totalConnections / entities.length : 0,
        creationRate,
        deletionRate,
        topEntities: topEntities.sort((a, b) => b.connectionCount - a.connectionCount),
        typeDistribution
      };

    } catch (error) {
      console.error('Error collecting entity operations:', error);
      return null;
    }
  }

  /**
   * Collects graph topology data
   */
  async collectGraphTopology(): Promise<TopologyData | null> {
    try {
      // Get graph structure data
      const graphData = await this.mcpClient.llmkg.knowledgeGraphQuery({
        query: 'CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }',
        limit: -1, // Get all triples for topology analysis
        includeWeights: true
      });

      if (!graphData?.graph) {
        return null;
      }

      const graph = graphData.graph;
      const nodeCount = graph.nodes?.length || 0;
      const edgeCount = graph.edges?.length || 0;

      // Calculate topology metrics
      const density = this.calculateGraphDensity(nodeCount, edgeCount);
      const components = this.analyzeConnectedComponents(graph);
      const clustering = this.calculateClusteringCoefficient(graph);
      const { diameter, avgPathLength } = this.calculatePathMetrics(graph);
      const degreeDistribution = this.calculateDegreeDistribution(graph);

      const topologyData: TopologyData = {
        nodeCount,
        edgeCount,
        density,
        connectedComponents: components,
        clusteringCoefficient: clustering,
        diameter,
        avgPathLength,
        degreeDistribution
      };

      this.topologyCache = topologyData;
      this.lastTopologyUpdate = Date.now();

      return topologyData;

    } catch (error) {
      console.error('Error collecting graph topology:', error);
      return null;
    }
  }

  /**
   * Collects triple relationship metrics
   */
  async collectTripleMetrics(): Promise<TripleMetrics | null> {
    try {
      // Query triple statistics
      const tripleData = await this.mcpClient.llmkg.knowledgeGraphQuery({
        query: 'SELECT ?predicate (COUNT(*) as ?count) WHERE { ?s ?predicate ?o } GROUP BY ?predicate ORDER BY DESC(?count)',
        limit: 1000
      });

      if (!tripleData?.triples) {
        return null;
      }

      const triples = tripleData.triples;
      const predicateCount = new Set(triples.map((t: any) => t.predicate)).size;
      
      // Calculate predicate statistics
      const predicateStats: Record<string, number> = {};
      const distributions = {
        subjects: {} as Record<string, number>,
        predicates: {} as Record<string, number>,
        objects: {} as Record<string, number>
      };

      for (const triple of triples) {
        predicateStats[triple.predicate] = (predicateStats[triple.predicate] || 0) + 1;
        distributions.subjects[triple.subject] = (distributions.subjects[triple.subject] || 0) + 1;
        distributions.predicates[triple.predicate] = (distributions.predicates[triple.predicate] || 0) + 1;
        distributions.objects[triple.object] = (distributions.objects[triple.object] || 0) + 1;
      }

      const topPredicates: PredicateInfo[] = Object.entries(predicateStats)
        .map(([predicate, count]) => ({
          predicate,
          count,
          percentage: (count / triples.length) * 100
        }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10);

      // Calculate rates
      const now = Date.now();
      const creationRate = this.calculateTripleCreationRate();
      const validationRate = this.calculateValidationRate();

      return {
        tripleCount: triples.length,
        predicateCount,
        topPredicates,
        creationRate,
        validationRate,
        distributions
      };

    } catch (error) {
      console.error('Error collecting triple metrics:', error);
      return null;
    }
  }

  /**
   * Collects query performance metrics
   */
  async collectQueryMetrics(): Promise<QueryMetrics | null> {
    try {
      // Analyze query patterns from cache
      const queryPatterns: QueryPattern[] = Array.from(this.queryPatternCache.values())
        .sort((a, b) => b.count - a.count)
        .slice(0, 20);

      // Calculate complexity distribution
      const complexityDistribution: Record<string, number> = {
        simple: 0,
        medium: 0,
        complex: 0
      };

      for (const pattern of queryPatterns) {
        const complexity = this.analyzeQueryComplexity(pattern.pattern);
        complexityDistribution[complexity]++;
      }

      // Calculate average execution time
      const avgExecutionTime = this.queryExecutionTimes.length > 0
        ? this.queryExecutionTimes.reduce((a, b) => a + b, 0) / this.queryExecutionTimes.length
        : 0;

      // Calculate success rate
      const successRate = this.calculateQuerySuccessRate();

      // Calculate throughput
      const throughput = this.aggregator.getRate(60000); // Queries per second over last minute

      // Estimate cache hit rate (simplified)
      const cacheHitRate = this.estimateCacheHitRate();

      return {
        queryCount: this.queryPatternCache.size,
        avgExecutionTime,
        successRate,
        queryPatterns,
        complexityDistribution,
        cacheHitRate,
        throughput
      };

    } catch (error) {
      console.error('Error collecting query metrics:', error);
      return null;
    }
  }

  /**
   * Sets up event handlers for graph operations
   */
  private setupGraphEventHandlers(): void {
    this.mcpClient.on('mcp:tool:response', (event) => {
      if (event.data.toolName.includes('knowledge_graph')) {
        this.processGraphToolResponse(event);
      }
    });

    // Monitor query executions
    this.on('mcp:tool:response', (event) => {
      if (event.data.toolName === 'knowledge_graph_query') {
        this.recordQueryExecution(event);
      }
    });
  }

  /**
   * Collects initial graph state
   */
  private async collectInitialState(): Promise<void> {
    try {
      // Get basic graph statistics
      const stats = await this.mcpClient.llmkg.knowledgeGraphQuery({
        query: 'SELECT (COUNT(DISTINCT ?s) as ?subjects) (COUNT(DISTINCT ?p) as ?predicates) (COUNT(DISTINCT ?o) as ?objects) WHERE { ?s ?p ?o }',
        limit: 1
      });

      if (stats) {
        console.log(`Initial graph state: ${JSON.stringify(stats)}`);
      }
    } catch (error) {
      console.warn('Failed to collect initial graph state:', error);
    }
  }

  /**
   * Performs periodic topology analysis
   */
  private async performTopologyAnalysis(): Promise<void> {
    if (!this.isRunning()) return;

    try {
      const topologyData = await this.collectGraphTopology();
      if (topologyData) {
        this.emit('topology:updated', {
          collector: this.config.name,
          topology: topologyData,
          timestamp: Date.now()
        });
      }
    } catch (error) {
      console.error('Error in topology analysis:', error);
    }
  }

  /**
   * Processes graph tool responses for metrics
   */
  private processGraphToolResponse(event: any): void {
    const { toolName, result, duration } = event.data;
    
    // Record execution time
    if (duration) {
      this.queryExecutionTimes.push(duration);
      if (this.queryExecutionTimes.length > 1000) {
        this.queryExecutionTimes = this.queryExecutionTimes.slice(-1000);
      }
    }

    // Update query pattern cache
    if (toolName === 'knowledge_graph_query' && event.data.params?.query) {
      this.updateQueryPatternCache(event.data.params.query, duration || 0, !!result);
    }
  }

  /**
   * Records query execution for metrics
   */
  private recordQueryExecution(event: any): void {
    const { params, result, duration, error } = event.data;
    
    if (params?.query) {
      const pattern = this.normalizeQueryPattern(params.query);
      this.updateQueryPatternCache(pattern, duration || 0, !error);
    }
  }

  /**
   * Helper methods for calculations
   */
  private shouldUpdateTopology(): boolean {
    const interval = (this.config as KnowledgeGraphCollectorConfig).topologyInterval;
    return Date.now() - this.lastTopologyUpdate > interval;
  }

  private calculateGraphDensity(nodeCount: number, edgeCount: number): number {
    if (nodeCount <= 1) return 0;
    const maxPossibleEdges = nodeCount * (nodeCount - 1) / 2;
    return edgeCount / maxPossibleEdges;
  }

  private analyzeConnectedComponents(graph: any): number {
    // Simplified connected component analysis
    // In a real implementation, this would use a graph traversal algorithm
    return graph.components || 1;
  }

  private calculateClusteringCoefficient(graph: any): number {
    // Simplified clustering coefficient calculation
    return graph.clustering_coefficient || 0;
  }

  private calculatePathMetrics(graph: any): { diameter: number; avgPathLength: number } {
    // Simplified path metrics - would require graph algorithms in real implementation
    return {
      diameter: graph.diameter || 0,
      avgPathLength: graph.avg_path_length || 0
    };
  }

  private calculateDegreeDistribution(graph: any): DegreeStats {
    const degrees = graph.nodes?.map((node: any) => node.degree || 0) || [];
    
    if (degrees.length === 0) {
      return { min: 0, max: 0, mean: 0, median: 0, stdDev: 0, histogram: {} };
    }

    const sorted = degrees.sort((a: number, b: number) => a - b);
    const sum = degrees.reduce((a: number, b: number) => a + b, 0);
    const mean = sum / degrees.length;
    const variance = degrees.reduce((acc: number, val: number) => acc + Math.pow(val - mean, 2), 0) / degrees.length;
    
    const histogram: Record<number, number> = {};
    for (const degree of degrees) {
      histogram[degree] = (histogram[degree] || 0) + 1;
    }

    return {
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean,
      median: sorted[Math.floor(sorted.length / 2)],
      stdDev: Math.sqrt(variance),
      histogram
    };
  }

  private calculateCreationRate(type: string, now: number): number {
    // Simplified rate calculation - would track actual creation events in practice
    return this.aggregator.getRate(60000);
  }

  private calculateDeletionRate(type: string, now: number): number {
    // Simplified rate calculation
    return 0;
  }

  private calculateTripleCreationRate(): number {
    if (this.tripleCreationTimes.length === 0) return 0;
    const now = Date.now();
    const recent = this.tripleCreationTimes.filter(time => now - time < 60000);
    return recent.length / 60; // Per second over last minute
  }

  private calculateValidationRate(): number {
    // Placeholder - would track validation success/failure in practice
    return 0.95;
  }

  private calculateQuerySuccessRate(): number {
    let totalQueries = 0;
    let successfulQueries = 0;

    for (const pattern of this.queryPatternCache.values()) {
      totalQueries += pattern.count;
      successfulQueries += Math.floor(pattern.count * pattern.successRate);
    }

    return totalQueries > 0 ? successfulQueries / totalQueries : 1.0;
  }

  private estimateCacheHitRate(): number {
    // Simplified cache hit rate estimation
    return 0.75;
  }

  private analyzeQueryComplexity(query: string): string {
    const joins = (query.match(/JOIN/gi) || []).length;
    const unions = (query.match(/UNION/gi) || []).length;
    const subqueries = (query.match(/\{[^}]*\{/g) || []).length;
    
    const complexity = joins + unions * 2 + subqueries * 3;
    
    if (complexity <= 2) return 'simple';
    if (complexity <= 6) return 'medium';
    return 'complex';
  }

  private updateQueryPatternCache(query: string, duration: number, success: boolean): void {
    const pattern = this.normalizeQueryPattern(query);
    
    if (this.queryPatternCache.has(pattern)) {
      const existing = this.queryPatternCache.get(pattern)!;
      existing.count++;
      existing.avgExecutionTime = (existing.avgExecutionTime * (existing.count - 1) + duration) / existing.count;
      existing.successRate = (existing.successRate * (existing.count - 1) + (success ? 1 : 0)) / existing.count;
    } else {
      if (this.queryPatternCache.size >= (this.config as KnowledgeGraphCollectorConfig).queryPatternCacheSize) {
        // Remove least used pattern
        const entries = Array.from(this.queryPatternCache.entries());
        const leastUsed = entries.reduce((min, [key, value]) => 
          value.count < min[1].count ? [key, value] : min
        );
        this.queryPatternCache.delete(leastUsed[0]);
      }

      this.queryPatternCache.set(pattern, {
        pattern,
        count: 1,
        avgExecutionTime: duration,
        successRate: success ? 1.0 : 0.0
      });
    }
  }

  private normalizeQueryPattern(query: string): string {
    // Normalize query by removing specific values and standardizing format
    return query
      .replace(/["'][^"']*["']/g, '"VALUE"')
      .replace(/\b\d+\b/g, 'NUM')
      .replace(/\s+/g, ' ')
      .trim()
      .toUpperCase();
  }

  private createCollectedData(type: string, data: any, method: string): CollectedData {
    return {
      id: this.generateId(),
      timestamp: Date.now(),
      source: 'knowledge-graph',
      type,
      data,
      metadata: this.createMetadata(method, undefined, { graphType: type })
    };
  }
}