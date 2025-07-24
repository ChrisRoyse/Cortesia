/**
 * Service for fetching ALL real data from the LLMKG knowledge graph
 * Connects to actual backend APIs and MCP tools to retrieve entities, relationships, and database content
 */

import { MCPTool } from '../types';

// Backend API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api/v1';
const WEBSOCKET_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8081';

// Real knowledge graph data types matching backend
export interface RealTriple {
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  source?: string;
  metadata?: Record<string, any>;
  timestamp?: number;
}

export interface RealEntity {
  id: string;
  name: string;
  entity_type: string;
  description: string;
  properties: Record<string, any>;
  embedding?: number[];
  created_at?: number;
  updated_at?: number;
}

export interface RealRelationship {
  source: string;
  target: string;
  relationship_type: string;
  weight: number;
  metadata?: Record<string, any>;
  confidence?: number;
}

export interface RealChunk {
  id: string;
  text: string;
  embedding?: number[];
  metadata?: Record<string, any>;
  created_at?: number;
}

export interface DatabaseInfo {
  name: string;
  type: string;
  entity_count: number;
  triple_count: number;
  chunk_count: number;
  size_bytes: number;
  last_updated: number;
}

export interface KnowledgeGraphStats {
  total_entities: number;
  total_triples: number;
  total_chunks: number;
  total_relationships: number;
  entity_types: Record<string, number>;
  predicate_types: Record<string, number>;
  databases: DatabaseInfo[];
  memory_usage: number;
  query_performance: {
    avg_query_time_ms: number;
    total_queries: number;
    cache_hit_rate: number;
  };
}

export interface ComprehensiveKnowledgeGraphData {
  entities: RealEntity[];
  triples: RealTriple[];
  relationships: RealRelationship[];
  chunks: RealChunk[];
  stats: KnowledgeGraphStats;
  databases: DatabaseInfo[];
  entity_types: string[];
  predicate_suggestions: string[];
  last_updated: number;
}

export class KnowledgeGraphDataService {
  private cache: Map<string, any> = new Map();
  private cacheExpiry: Map<string, number> = new Map();
  private readonly CACHE_DURATION = 30000; // 30 seconds

  /**
   * Fetch ALL entities from the knowledge graph
   */
  async fetchAllEntities(): Promise<RealEntity[]> {
    const cacheKey = 'all_entities';
    if (this.isCacheValid(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      // Try MCP tool first
      const response = await this.executeMCPTool('hybrid_search', {
        query: '*',
        search_type: 'structural',
        limit: 10000,
        filters: { entity_types: [] }
      });

      if (response && response.entities && response.entities.length > 0) {
        this.setCache(cacheKey, response.entities);
        return response.entities;
      }
    } catch (error) {
      console.warn('MCP tool failed, using fallback:', error);
    }

    // Fallback: generate entities from triples data
    return this.fetchEntitiesViaAPI();
  }

  /**
   * Fetch ALL triples from the knowledge graph
   */
  async fetchAllTriples(): Promise<RealTriple[]> {
    const cacheKey = 'all_triples';
    if (this.isCacheValid(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      // Query for all triples using pattern matching
      const response = await this.apiRequest('/query', {
        method: 'POST',
        body: JSON.stringify({
          subject: null,
          predicate: null,
          object: null,
          limit: 50000 // Large limit to get all triples
        })
      });

      const triples = response.data?.triples || [];
      this.setCache(cacheKey, triples);
      return triples;
    } catch (error) {
      console.error('Error fetching triples:', error);
      return [];
    }
  }

  /**
   * Fetch ALL relationships between entities
   */
  async fetchAllRelationships(): Promise<RealRelationship[]> {
    const cacheKey = 'all_relationships';
    if (this.isCacheValid(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      // Try MCP tool first
      const response = await this.executeMCPTool('explore_connections', {
        start_entity: '*',
        max_depth: 5,
        relationship_types: []
      });

      if (response && response.relationships && response.relationships.length > 0) {
        this.setCache(cacheKey, response.relationships);
        return response.relationships;
      }
    } catch (error) {
      console.warn('MCP relationships tool failed, generating from triples:', error);
    }

    // Fallback: generate relationships from triples
    return this.generateRelationshipsFromTriples();
  }

  /**
   * Generate relationships from triples data
   */
  private async generateRelationshipsFromTriples(): Promise<RealRelationship[]> {
    try {
      const triples = await this.fetchAllTriples();
      const relationships: RealRelationship[] = [];

      triples.forEach((triple: RealTriple) => {
        // Skip simple type relationships
        if (triple.predicate === 'is' || triple.predicate === 'type') {
          return;
        }

        relationships.push({
          source: triple.subject,
          target: triple.object,
          relationship_type: triple.predicate,
          weight: triple.confidence || 0.5,
          confidence: triple.confidence,
          metadata: triple.metadata
        });
      });

      console.log(`🔄 Generated ${relationships.length} relationships from triples`);
      return relationships;
    } catch (error) {
      console.error('Error generating relationships from triples:', error);
      return [];
    }
  }

  /**
   * Fetch ALL text chunks
   */
  async fetchAllChunks(): Promise<RealChunk[]> {
    const cacheKey = 'all_chunks';
    if (this.isCacheValid(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      // Use semantic search with wildcard to get all chunks
      const response = await this.apiRequest('/search', {
        method: 'POST',
        body: JSON.stringify({
          query: '',
          limit: 10000,
          threshold: 0.0 // Include all chunks regardless of similarity
        })
      });

      const chunks = response.data?.chunks || [];
      this.setCache(cacheKey, chunks);
      return chunks;
    } catch (error) {
      console.error('Error fetching chunks:', error);
      return [];
    }
  }

  /**
   * Fetch comprehensive knowledge graph statistics
   */
  async fetchKnowledgeGraphStats(): Promise<KnowledgeGraphStats> {
    const cacheKey = 'kg_stats';
    if (this.isCacheValid(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      // Try MCP tool first
      const statsResponse = await this.executeMCPTool('get_stats', {
        include_details: true
      });

      if (statsResponse) {
        const stats: KnowledgeGraphStats = {
          total_entities: statsResponse.entity_count || 0,
          total_triples: statsResponse.triple_count || 0,
          total_chunks: statsResponse.chunk_count || 0,
          total_relationships: statsResponse.relationship_count || 0,
          entity_types: statsResponse.entity_types || {},
          predicate_types: statsResponse.predicate_types || {},
          databases: [],
          memory_usage: 0,
          query_performance: {
            avg_query_time_ms: 0,
            total_queries: 0,
            cache_hit_rate: 0
          }
        };

        this.setCache(cacheKey, stats);
        return stats;
      }
    } catch (error) {
      console.warn('MCP stats tool failed, generating basic stats:', error);
    }

    // Fallback: generate stats from actual data
    return this.generateBasicStats();
  }

  /**
   * Generate basic statistics from current data
   */
  private async generateBasicStats(): Promise<KnowledgeGraphStats> {
    try {
      const [entities, triples, relationships, chunks] = await Promise.all([
        this.fetchAllEntities(),
        this.fetchAllTriples(),
        this.fetchAllRelationships(),
        this.fetchAllChunks()
      ]);

      // Calculate entity types
      const entityTypes: Record<string, number> = {};
      entities.forEach(entity => {
        entityTypes[entity.entity_type] = (entityTypes[entity.entity_type] || 0) + 1;
      });

      // Calculate predicate types
      const predicateTypes: Record<string, number> = {};
      triples.forEach(triple => {
        predicateTypes[triple.predicate] = (predicateTypes[triple.predicate] || 0) + 1;
      });

      const stats: KnowledgeGraphStats = {
        total_entities: entities.length,
        total_triples: triples.length,
        total_chunks: chunks.length,
        total_relationships: relationships.length,
        entity_types: entityTypes,
        predicate_types: predicateTypes,
        databases: [{
          name: 'Main Knowledge Graph',
          type: 'In-Memory Graph',
          entity_count: entities.length,
          triple_count: triples.length,
          chunk_count: chunks.length,
          size_bytes: 0,
          last_updated: Date.now()
        }],
        memory_usage: 0,
        query_performance: {
          avg_query_time_ms: 0,
          total_queries: 0,
          cache_hit_rate: 0
        }
      };

      console.log('📊 Generated basic stats:', stats);
      return stats;
    } catch (error) {
      console.error('Error generating basic stats:', error);
      return this.getDefaultStats();
    }
  }

  /**
   * Fetch available entity types
   */
  async fetchEntityTypes(): Promise<string[]> {
    try {
      const response = await this.apiRequest('/entity-types', {
        method: 'GET'
      });
      return response.data?.entity_types || [];
    } catch (error) {
      console.error('Error fetching entity types:', error);
      return [];
    }
  }

  /**
   * Get predicate suggestions for LLM context
   */
  async fetchPredicateSuggestions(context: string = ''): Promise<string[]> {
    try {
      const response = await this.apiRequest(`/suggest-predicates?context=${encodeURIComponent(context)}`, {
        method: 'GET'
      });
      return response.data?.predicates || [];
    } catch (error) {
      console.error('Error fetching predicate suggestions:', error);
      return [];
    }
  }

  /**
   * Fetch ALL knowledge graph data in one comprehensive call
   */
  async fetchComprehensiveData(): Promise<ComprehensiveKnowledgeGraphData> {
    console.log('🔍 Fetching comprehensive knowledge graph data...');
    
    // Fetch all data in parallel for performance
    const [entities, triples, relationships, chunks, stats, entityTypes, predicates] = await Promise.all([
      this.fetchAllEntities(),
      this.fetchAllTriples(),
      this.fetchAllRelationships(),
      this.fetchAllChunks(),
      this.fetchKnowledgeGraphStats(),
      this.fetchEntityTypes(),
      this.fetchPredicateSuggestions()
    ]);

    const data: ComprehensiveKnowledgeGraphData = {
      entities,
      triples,
      relationships,
      chunks,
      stats,
      databases: stats.databases,
      entity_types: entityTypes,
      predicate_suggestions: predicates,
      last_updated: Date.now()
    };

    console.log('✅ Comprehensive data fetched:', {
      entities: entities.length,
      triples: triples.length,
      relationships: relationships.length,
      chunks: chunks.length,
      databases: stats.databases.length
    });

    return data;
  }

  /**
   * Execute MCP tool with parameters
   */
  async executeMCPTool(toolName: string, parameters: Record<string, any>): Promise<any> {
    try {
      // Try the MCP endpoint first
      const response = await fetch(`http://localhost:3001/mcp/tools/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          tool: `mcp__llmkg__${toolName}`,
          parameters
        })
      });

      if (!response.ok) {
        throw new Error(`MCP tool ${toolName} failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result.result || result.data || result;
    } catch (error) {
      console.error(`Error executing MCP tool ${toolName}:`, error);
      throw error;
    }
  }

  /**
   * Generic API request helper
   */
  private async apiRequest(endpoint: string, options: RequestInit = {}): Promise<any> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Fallback method to fetch entities via REST API
   */
  private async fetchEntitiesViaAPI(): Promise<RealEntity[]> {
    try {
      // Get all triples to extract unique subjects as entities
      const response = await this.apiRequest('/query', {
        method: 'POST',
        body: JSON.stringify({
          subject: null,
          predicate: null,
          object: null,
          limit: 10000
        })
      });

      const triples = response.data?.triples || [];
      const entityMap = new Map<string, RealEntity>();

      // Build entities from all subjects in triples
      triples.forEach((triple: RealTriple) => {
        // Add subject as entity
        if (!entityMap.has(triple.subject)) {
          entityMap.set(triple.subject, {
            id: triple.subject,
            name: triple.subject,
            entity_type: 'unknown',
            description: '',
            properties: {},
            created_at: Date.now()
          });
        }

        // Update entity properties based on predicate
        const entity = entityMap.get(triple.subject)!;
        if (triple.predicate === 'is' || triple.predicate === 'type') {
          entity.entity_type = triple.object;
        } else if (triple.predicate === 'name') {
          entity.name = triple.object;
        } else if (triple.predicate === 'description') {
          entity.description = triple.object;
        } else {
          entity.properties[triple.predicate] = triple.object;
        }
      });

      const entities = Array.from(entityMap.values());
      console.log(`🔄 Extracted ${entities.length} entities from ${triples.length} triples`);
      return entities;
    } catch (error) {
      console.error('Error fetching entities via API:', error);
      return [];
    }
  }

  /**
   * Extract database information from metrics
   */
  private extractDatabaseInfo(metrics: any): DatabaseInfo[] {
    const databases: DatabaseInfo[] = [];
    
    if (metrics.databases) {
      metrics.databases.forEach((db: any) => {
        databases.push({
          name: db.name || 'Main Knowledge Graph',
          type: db.type || 'In-Memory Graph',
          entity_count: db.entity_count || 0,
          triple_count: db.triple_count || 0,
          chunk_count: db.chunk_count || 0,
          size_bytes: db.size_bytes || 0,
          last_updated: db.last_updated || Date.now()
        });
      });
    } else {
      // Default database info
      databases.push({
        name: 'Main Knowledge Graph',
        type: 'In-Memory Graph',
        entity_count: metrics.entity_count || 0,
        triple_count: metrics.triple_count || 0,
        chunk_count: metrics.chunk_count || 0,
        size_bytes: metrics.memory_usage_bytes || 0,
        last_updated: Date.now()
      });
    }

    return databases;
  }

  /**
   * Get default stats when API fails
   */
  private getDefaultStats(): KnowledgeGraphStats {
    return {
      total_entities: 0,
      total_triples: 0,
      total_chunks: 0,
      total_relationships: 0,
      entity_types: {},
      predicate_types: {},
      databases: [],
      memory_usage: 0,
      query_performance: {
        avg_query_time_ms: 0,
        total_queries: 0,
        cache_hit_rate: 0
      }
    };
  }

  /**
   * Cache management
   */
  private isCacheValid(key: string): boolean {
    const expiry = this.cacheExpiry.get(key);
    return expiry ? Date.now() < expiry : false;
  }

  private setCache(key: string, value: any): void {
    this.cache.set(key, value);
    this.cacheExpiry.set(key, Date.now() + this.CACHE_DURATION);
  }

  /**
   * Clear all cached data
   */
  clearCache(): void {
    this.cache.clear();
    this.cacheExpiry.clear();
  }

  /**
   * Search entities by query
   */
  async searchEntities(query: string, limit: number = 100): Promise<RealEntity[]> {
    try {
      const response = await this.executeMCPTool('hybrid_search', {
        query,
        search_type: 'hybrid',
        limit,
        filters: {}
      });

      return response.entities || [];
    } catch (error) {
      console.error('Error searching entities:', error);
      return [];
    }
  }

  /**
   * Get entity details and all its relationships
   */
  async getEntityDetails(entityId: string): Promise<{
    entity: RealEntity | null;
    triples: RealTriple[];
    relationships: RealRelationship[];
  }> {
    try {
      const [triplesResponse, relationshipsResponse] = await Promise.all([
        this.apiRequest('/query', {
          method: 'POST',
          body: JSON.stringify({
            subject: entityId,
            limit: 1000
          })
        }),
        this.executeMCPTool('explore_connections', {
          start_entity: entityId,
          max_depth: 2
        })
      ]);

      const triples = triplesResponse.data?.triples || [];
      const relationships = relationshipsResponse.relationships || [];
      
      // Extract entity from triples
      const entity = this.extractEntityFromTriples(entityId, triples);

      return {
        entity,
        triples,
        relationships
      };
    } catch (error) {
      console.error('Error getting entity details:', error);
      return {
        entity: null,
        triples: [],
        relationships: []
      };
    }
  }

  private extractEntityFromTriples(entityId: string, triples: RealTriple[]): RealEntity | null {
    const properties: Record<string, any> = {};
    let entityType = 'unknown';
    let name = entityId;
    let description = '';

    triples.forEach(triple => {
      if (triple.subject === entityId) {
        if (triple.predicate === 'type') {
          entityType = triple.object;
        } else if (triple.predicate === 'name') {
          name = triple.object;
        } else if (triple.predicate === 'description') {
          description = triple.object;
        } else {
          properties[triple.predicate] = triple.object;
        }
      }
    });

    return {
      id: entityId,
      name,
      entity_type: entityType,
      description,
      properties,
      created_at: Date.now()
    };
  }
}

// Export singleton instance
export const knowledgeGraphDataService = new KnowledgeGraphDataService();