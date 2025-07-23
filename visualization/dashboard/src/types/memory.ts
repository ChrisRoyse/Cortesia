export interface SDRStorage {
  totalSDRs: number;
  activeSDRs: number;
  archivedSDRs: number;
  totalMemoryBytes: number;
  averageSparsity: number;
  compressionRatio: number;
  fragmentationLevel: number;
  storageBlocks: SDRStorageBlock[];
}

export interface SDRStorageBlock {
  id: string;
  size: number;
  used: number;
  fragmented: number;
  patterns: number;
  lastAccess: number;
  compressionType: 'overlap' | 'dictionary' | 'none';
}

export interface KnowledgeGraphMemory {
  entities: MemoryBlock;
  relations: MemoryBlock;
  embeddings: MemoryBlock;
  indexes: MemoryBlock;
  cache: MemoryBlock;
}

export interface MemoryBlock {
  name: string;
  size: number;
  used: number;
  value?: number;
  children?: MemoryBlock[];
  metadata?: {
    accessCount: number;
    lastAccess: number;
    fragmentation: number;
  };
}

export interface HierarchicalMemoryData {
  name: string;
  size: number;
  value?: number;
  children?: HierarchicalMemoryData[];
}

export interface ZeroCopyMetrics {
  enabled: boolean;
  totalOperations: number;
  savedBytes: number;
  copyOnWriteEvents: number;
  sharedRegions: number;
  efficiency: number;
}

export interface MemoryFlow {
  timestamp: number;
  source: string;
  target: string;
  bytes: number;
  operation: 'allocate' | 'free' | 'copy' | 'share';
  duration: number;
}

export interface CognitiveLayerMemory {
  subcortical: {
    total: number;
    used: number;
    components: {
      thalamus: number;
      hippocampus: number;
      amygdala: number;
      basalGanglia: number;
    };
  };
  cortical: {
    total: number;
    used: number;
    regions: {
      prefrontal: number;
      temporal: number;
      parietal: number;
      occipital: number;
    };
  };
  workingMemory: {
    capacity: number;
    used: number;
    buffers: WorkingMemoryBuffer[];
  };
}

export interface WorkingMemoryBuffer {
  id: string;
  content: string;
  size: number;
  age: number;
  accessCount: number;
  priority: number;
}

export interface MemoryPressure {
  level: 'low' | 'medium' | 'high' | 'critical';
  percentage: number;
  swapUsed: number;
  pageCache: number;
  recommendations: string[];
}