// API Service for LLMKG Dashboard
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001/api/v1';

export interface ApiResponse<T> {
  status: 'success' | 'error';
  data?: T;
  error?: string;
}

export interface StoreTripleRequest {
  subject: string;
  predicate: string;
  object: string;
  confidence?: number;
}

export interface StoreTripleResponse {
  node_id: string;
}

export interface QueryTriplesRequest {
  subject?: string;
  predicate?: string;
  object?: string;
  limit?: number;
}

export interface Triple {
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  source?: string;
}

export interface QueryResponse {
  triples: Triple[];
  chunks: any[];
  query_time_ms: number;
}

export interface MetricsResponse {
  entity_count: number;
  memory_stats: {
    total_nodes: number;
    total_triples: number;
    total_bytes: number;
    bytes_per_node: number;
    cache_hits: number;
    cache_misses: number;
  };
  entity_types: Record<string, string>;
}

export interface ApiEndpoint {
  path: string;
  method: string;
  description: string;
  request_schema?: any;
  response_schema?: any;
}

export interface ApiDiscoveryResponse {
  version: string;
  endpoints: ApiEndpoint[];
}

class ApiService {
  private async request<T>(path: string, options?: RequestInit): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${API_BASE_URL}${path}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      return {
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // Discovery
  async getApiDiscovery(): Promise<ApiResponse<ApiDiscoveryResponse>> {
    return this.request<ApiDiscoveryResponse>('/discovery');
  }

  // Storage operations
  async storeTriple(triple: StoreTripleRequest): Promise<ApiResponse<StoreTripleResponse>> {
    return this.request<StoreTripleResponse>('/triple', {
      method: 'POST',
      body: JSON.stringify(triple),
    });
  }

  async storeChunk(text: string, embedding?: number[]): Promise<ApiResponse<StoreTripleResponse>> {
    return this.request<StoreTripleResponse>('/chunk', {
      method: 'POST',
      body: JSON.stringify({ text, embedding }),
    });
  }

  async storeEntity(entity: {
    name: string;
    entity_type: string;
    description: string;
    properties: Record<string, string>;
  }): Promise<ApiResponse<StoreTripleResponse>> {
    return this.request<StoreTripleResponse>('/entity', {
      method: 'POST',
      body: JSON.stringify(entity),
    });
  }

  // Query operations
  async queryTriples(query: QueryTriplesRequest): Promise<ApiResponse<QueryResponse>> {
    return this.request<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify(query),
    });
  }

  async semanticSearch(query: string, limit: number = 10): Promise<ApiResponse<any>> {
    return this.request<any>('/search', {
      method: 'POST',
      body: JSON.stringify({ query, limit }),
    });
  }

  async getEntityRelationships(entityName: string, maxHops?: number): Promise<ApiResponse<any>> {
    return this.request<any>('/relationships', {
      method: 'POST',
      body: JSON.stringify({ entity_name: entityName, max_hops: maxHops }),
    });
  }

  // Metrics
  async getMetrics(): Promise<ApiResponse<MetricsResponse>> {
    return this.request<MetricsResponse>('/metrics');
  }

  async getEntityTypes(): Promise<ApiResponse<Record<string, string>>> {
    return this.request<Record<string, string>>('/entity-types');
  }

  async suggestPredicates(context: string): Promise<ApiResponse<string[]>> {
    return this.request<string[]>(`/suggest-predicates?context=${encodeURIComponent(context)}`);
  }

  // Monitoring metrics
  async getMonitoringMetrics(): Promise<ApiResponse<any>> {
    return this.request<any>('/monitoring/metrics');
  }
}

export const apiService = new ApiService();