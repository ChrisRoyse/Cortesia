function openTab(evt, tabName) {
  // Declare all variables
  var i, tabcontent, tablinks;

  // Get all elements with class="tabcontent" and hide them
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  // Get all elements with class="tablinks" and remove the class "active"
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }

  // Show the current tab, and add an "active" class to the button that opened the tab
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}

// Main visualization
const mainNodes = new vis.DataSet([
  // High-level modules
  { id: 1, label: 'Core', group: 'main' },
  { id: 2, label: 'Embedding', group: 'main' },
  { id: 3, label: 'Storage', group: 'main' },
  { id: 4, label: 'Query', group: 'main' },
  { id: 5, label: 'Cognitive', group: 'main' },
  { id: 6, label: 'Learning', group: 'main' },
  { id: 7, label: 'Federation', group: 'main' },
  { id: 8, label: 'WASM', group: 'main' },
  { id: 9, label: 'Neural', group: 'main' },
  { id: 10, label: 'Extraction', group: 'main' },

  // Core sub-modules
  { id: 101, label: 'Graph', group: 'core' },
  { id: 102, label: 'Entity', group: 'core' },
  { id: 103, label: 'Knowledge Engine', group: 'core' },
  { id: 104, label: 'Activation Engine', group: 'core' },
  { id: 105, label: 'SDR', group: 'core' },

  // Knowledge Engine components
  { id: 106, label: 'KnowledgeEngine', group: 'core-engine' },
  { id: 1061, label: 'Nodes (HashMap)', group: 'core-engine-storage' },
  { id: 1062, label: 'Subject Index', group: 'core-engine-index' },
  { id: 1063, label: 'Predicate Index', group: 'core-engine-index' },
  { id: 1064, label: 'Object Index', group: 'core-engine-index' },
  { id: 1065, label: 'Entity Types', group: 'core-engine-metadata' },
  { id: 1066, label: 'Predicate Vocabulary', group: 'core-engine-metadata' },
  { id: 1067, label: 'Batch Processor', group: 'core-engine-embedding' },
  { id: 1068, label: 'Embedding Generator', group: 'core-engine-embedding' },
  { id: 1069, label: 'Triple Extractor', group: 'core-engine-extraction' },
  { id: 1070, label: 'Memory Stats', group: 'core-engine-monitoring' },
  { id: 1071, label: 'Frequent Patterns', group: 'core-engine-metadata' },

  // Core Data Structures
  { id: 201, label: 'EntityKey', group: 'core-data' },
  { id: 202, label: 'AttributeValue', group: 'core-data' },
  { id: 203, label: 'RelationshipType', group: 'core-data' },
  { id: 204, label: 'Weight', group: 'core-data' },
  { id: 205, label: 'EntityMeta', group: 'core-data' },
  { id: 206, label: 'EntityData', group: 'core-data' },
  { id: 207, label: 'Relationship', group: 'core-data' },
  { id: 208, label: 'ContextEntity', group: 'core-data' },
  { id: 209, label: 'QueryResult', group: 'core-data' },
  { id: 210, label: 'CompactEntity', group: 'core-data' },
  { id: 211, label: 'NeighborSlice', group: 'core-data' },
  { id: 212, label: 'SIMDRelationship', group: 'core-data' },
  { id: 213, label: 'QueryParams', group: 'core-data' },
  { id: 214, label: 'GraphQuery', group: 'core-data' },
  { id: 215, label: 'TraversalParams', group: 'core-data' },

  // Core Data Structures
  { id: 201, label: 'EntityKey', group: 'core-data' },
  { id: 202, label: 'AttributeValue', group: 'core-data' },
  { id: 203, label: 'RelationshipType', group: 'core-data' },
  { id: 204, label: 'Weight', group: 'core-data' },
  { id: 205, label: 'EntityMeta', group: 'core-data' },
  { id: 206, label: 'EntityData', group: 'core-data' },
  { id: 207, label: 'Relationship', group: 'core-data' },
  { id: 208, label: 'ContextEntity', group: 'core-data' },
  { id: 209, label: 'QueryResult', group: 'core-data' },
  { id: 210, label: 'CompactEntity', group: 'core-data' },
  { id: 211, label: 'NeighborSlice', group: 'core-data' },
  { id: 212, label: 'SIMDRelationship', group: 'core-data' },
  { id: 213, label: 'QueryParams', group: 'core-data' },
  { id: 214, label: 'GraphQuery', group: 'core-data' },
  { id: 215, label: 'TraversalParams', group: 'core-data' },

  // Embedding sub-modules
  { id: 201, label: 'Quantizer', group: 'embedding' },
  { id: 202, label: 'Store', group: 'embedding' },
  { id: 203, label: 'Similarity', group: 'embedding' },
  { id: 204, label: 'SIMD Search', group: 'embedding' },

  // Storage sub-modules
  { id: 301, label: 'Persistent MMap', group: 'storage' },
  { id: 302, label: 'String Interner', group: 'storage' },
  { id: 303, label: 'HNSW', group: 'storage' },
  { id: 304, label: 'LSH', group: 'storage' },

  // Query sub-modules
  { id: 401, label: 'RAG', group: 'query' },
  { id: 402, label: 'Optimizer', group: 'query' },
  { id: 403, label: 'Clustering', group: 'query' },
  { id: 404, label: 'Summarization', group: 'query' },

  // Cognitive sub-modules
  { id: 501, label: 'Orchestrator', group: 'cognitive' },
  { id: 502, label: 'Convergent Thinking', group: 'cognitive' },
  { id: 503, label: 'Divergent Thinking', group: 'cognitive' },
  { id: 504, label: 'Lateral Thinking', group: 'cognitive' },
  { id: 505, label: 'Systems Thinking', group: 'cognitive' },
  { id: 506, label: 'Critical Thinking', group: 'cognitive' },
  { id: 507, label: 'Abstract Thinking', group: 'cognitive' },
  { id: 508, label: 'Adaptive Thinking', group: 'cognitive' },
  { id: 509, label: 'Attention Manager', group: 'cognitive' },
  { id: 510, label: 'Working Memory', group: 'cognitive' },

  // Learning sub-modules
  { id: 601, label: 'Hebbian Learning', group: 'learning' },
  { id: 602, label: 'Synaptic Homeostasis', group: 'learning' },
  { id: 603, label: 'Adaptive Learning', group: 'learning' },
  { id: 604, label: 'Meta Learning', group: 'learning' },

  // Federation sub-modules
  { id: 701, label: 'Registry', group: 'federation' },
  { id: 702, label: 'Router', group: 'federation' },
  { id: 703, label: 'Merger', group: 'federation' },
  { id: 704, label: 'Coordinator', group: 'federation' },

  // Neural sub-modules
  { id: 901, label: 'Summarization', group: 'neural' },
  { id: 902, label: 'Canonicalization', group: 'neural' },
  { id: 903, label: 'Salience', group: 'neural' },
  { id: 904, label: 'Structure Predictor', group: 'neural' },

  // Extraction sub-modules
  { id: 1001, label: 'Advanced NLP', group: 'extraction' },
]);

const mainEdges = new vis.DataSet([
  // High-level connections
  { from: 1, to: 2, label: 'uses' },
  { from: 1, to: 3, label: 'uses' },
  { from: 1, to: 4, label: 'uses' },
  { from: 1, to: 5, label: 'uses' },
  { from: 1, to: 6, label: 'uses' },
  { from: 5, to: 6, label: 'informs' },
  { from: 4, to: 7, label: 'enables' },
  { from: 1, to: 8, label: 'exposes' },
  { from: 1, to: 9, label: 'integrates' },
  { from: 1, to: 10, label: 'uses' },

  // Core sub-module connections
  { from: 1, to: 101, label: 'contains' },
  { from: 1, to: 102, label: 'contains' },
  { from: 1, to: 103, label: 'contains' },
  { from: 1, to: 104, label: 'contains' },
  { from: 1, to: 105, label: 'contains' },

  // Knowledge Engine connections
  { from: 1, to: 106, label: 'contains' },
  { from: 106, to: 1061, label: 'manages' },
  { from: 106, to: 1062, label: 'uses' },
  { from: 106, to: 1063, label: 'uses' },
  { from: 106, to: 1064, label: 'uses' },
  { from: 106, to: 1065, label: 'uses' },
  { from: 106, to: 1066, label: 'uses' },
  { from: 106, to: 1067, label: 'uses' },
  { from: 106, to: 1068, label: 'uses' },
  { from: 106, to: 1069, label: 'uses' },
  { from: 106, to: 1070, label: 'uses' },
  { from: 106, to: 1071, label: 'uses' },

  // Data flow within KnowledgeEngine
  { from: 1068, to: 1061, label: 'generates embeddings for' },
  { from: 1069, to: 1061, label: 'extracts triples for' },
  { from: 1061, to: 1062, label: 'indexes' },
  { from: 1061, to: 1063, label: 'indexes' },
  { from: 1061, to: 1064, label: 'indexes' },
  { from: 1062, to: 106, label: 'informs query' },
  { from: 1063, to: 106, label: 'informs query' },
  { from: 1064, to: 106, label: 'informs query' },
  { from: 106, to: 1070, label: 'updates' },
  { from: 106, to: 1071, label: 'updates' },

  // Connections to other modules
  { from: 106, to: 2, label: 'uses embedding' }, // KnowledgeEngine uses Embedding module
  { from: 106, to: 3, label: 'uses storage' },  // KnowledgeEngine uses Storage module
  { from: 106, to: 4, label: 'supports query' }, // KnowledgeEngine supports Query module
  { from: 106, to: 10, label: 'uses extraction' }, // KnowledgeEngine uses Extraction module
]);

  // Embedding sub-module connections
  { from: 2, to: 201, label: 'contains' },
  { from: 2, to: 202, label: 'contains' },
  { from: 2, to: 203, label: 'contains' },
  { from: 2, to: 204, label: 'contains' },

  // Storage sub-module connections
  { from: 3, to: 301, label: 'contains' },
  { from: 3, to: 302, label: 'contains' },
  { from: 3, to: 303, label: 'contains' },
  { from: 3, to: 304, label: 'contains' },

  // Query sub-module connections
  { from: 4, to: 401, label: 'contains' },
  { from: 4, to: 402, label: 'contains' },
  { from: 4, to: 403, label: 'contains' },
  { from: 4, to: 404, label: 'contains' },

  // Cognitive sub-module connections
  { from: 5, to: 501, label: 'contains' },
  { from: 5, to: 502, label: 'contains' },
  { from: 5, to: 503, label: 'contains' },
  { from: 5, to: 504, label: 'contains' },
  { from: 5, to: 505, label: 'contains' },
  { from: 5, to: 506, label: 'contains' },
  { from: 5, to: 507, label: 'contains' },
  { from: 5, to: 508, label: 'contains' },
  { from: 5, to: 509, label: 'contains' },
  { from: 5, to: 510, label: 'contains' },

  // Learning sub-module connections
  { from: 6, to: 601, label: 'contains' },
  { from: 6, to: 602, label: 'contains' },
  { from: 6, to: 603, label: 'contains' },
  { from: 6, to: 604, label: 'contains' },

  // Federation sub-module connections
  { from: 7, to: 701, label: 'contains' },
  { from: 7, to: 702, label: 'contains' },
  { from: 7, to: 703, label: 'contains' },
  { from: 7, to: 704, label: 'contains' },

  // Neural sub-module connections
  { from: 9, to: 901, label: 'contains' },
  { from: 9, to: 902, label: 'contains' },
  { from: 9, to: 903, label: 'contains' },
  { from: 9, to: 904, label: 'contains' },

  // Extraction sub-module connections
  { from: 10, to: 1001, label: 'contains' },
]);

const mainContainer = document.getElementById('main-network');
const mainData = { nodes: mainNodes, edges: mainEdges };
const mainOptions = {
  layout: {
    hierarchical: {
      direction: "UD",
      sortMethod: "directed"
    }
  },
  edges: {
    arrows: 'to'
  },
  groups: {
    main: { color: { background: '#f44336', border: '#c62828' }, font: { color: '#ffffff' } },
    core: { color: { background: '#2196f3', border: '#1976d2' }, font: { color: '#ffffff' } },
    embedding: { color: { background: '#4caf50', border: '#388e3c' }, font: { color: '#ffffff' } },
    storage: { color: { background: '#ff9800', border: '#f57c00' }, font: { color: '#ffffff' } },
    query: { color: { background: '#9c27b0', border: '#7b1fa2' }, font: { color: '#ffffff' } },
    cognitive: { color: { background: '#673ab7', border: '#512da8' }, font: { color: '#ffffff' } },
    learning: { color: { background: '#009688', border: '#00796b' }, font: { color: '#ffffff' } },
    federation: { color: { background: '#795548', border: '#5d4037' }, font: { color: '#ffffff' } },
    wasm: { color: { background: '#607d8b', border: '#455a64' }, font: { color: '#ffffff' } },
    neural: { color: { background: '#e91e63', border: '#c2185b' }, font: { color: '#ffffff' } },
    extraction: { color: { background: '#00bcd4', border: '#0097a7' }, font: { color: '#ffffff' } },
  }
};
const mainNetwork = new vis.Network(mainContainer, mainData, mainOptions);

// Core visualization
const coreNodes = new vis.DataSet([
  { id: 1, label: 'KnowledgeGraph', group: 'core' },
  { id: 2, label: 'GraphArena', group: 'core-storage' },
  { id: 3, label: 'EntityStore', group: 'core-storage' },
  { id: 4, label: 'CSRGraph', group: 'core-storage' },
  { id: 5, label: 'EmbeddingBank', group: 'embedding' },
  { id: 6, label: 'ProductQuantizer', group: 'embedding' },
  { id: 7, label: 'BloomFilter', group: 'indexing' },
  { id: 8, label: 'EntityIdMap', group: 'indexing' },
  { id: 9, label: 'EpochManager', group: 'concurrency' },
  { id: 10, label: 'StringDictionary', group: 'metadata' },

  // Knowledge Engine components
  { id: 106, label: 'KnowledgeEngine', group: 'core-engine' },
  { id: 1061, label: 'Nodes (HashMap)', group: 'core-engine-storage' },
  { id: 1062, label: 'Subject Index', group: 'core-engine-index' },
  { id: 1063, label: 'Predicate Index', group: 'core-engine-index' },
  { id: 1064, label: 'Object Index', group: 'core-engine-index' },
  { id: 1065, label: 'Entity Types', group: 'core-engine-metadata' },
  { id: 1066, label: 'Predicate Vocabulary', group: 'core-engine-metadata' },
  { id: 1067, label: 'Batch Processor', group: 'core-engine-embedding' },
  { id: 1068, label: 'Embedding Generator', group: 'core-engine-embedding' },
  { id: 1069, label: 'Triple Extractor', group: 'core-engine-extraction' },
  { id: 1070, label: 'Memory Stats', group: 'core-engine-monitoring' },
  { id: 1071, label: 'Frequent Patterns', group: 'core-engine-metadata' },

  // Core Data Structures
  { id: 201, label: 'EntityKey', group: 'core-data' },
  { id: 202, label: 'AttributeValue', group: 'core-data' },
  { id: 203, label: 'RelationshipType', group: 'core-data' },
  { id: 204, label: 'Weight', group: 'core-data' },
  { id: 205, label: 'EntityMeta', group: 'core-data' },
  { id: 206, label: 'EntityData', group: 'core-data' },
  { id: 207, label: 'Relationship', group: 'core-data' },
  { id: 208, label: 'ContextEntity', group: 'core-data' },
  { id: 209, label: 'QueryResult', group: 'core-data' },
  { id: 210, label: 'CompactEntity', group: 'core-data' },
  { id: 211, label: 'NeighborSlice', group: 'core-data' },
  { id: 212, label: 'SIMDRelationship', group: 'core-data' },
  { id: 213, label: 'QueryParams', group: 'core-data' },
  { id: 214, label: 'GraphQuery', group: 'core-data' },
  { id: 215, label: 'TraversalParams', group: 'core-data' },
]);

const coreEdges = new vis.DataSet([
  { from: 1, to: 2, label: 'uses' },
  { from: 1, to: 3, label: 'uses' },
  { from: 1, to: 4, label: 'uses' },
  { from: 1, to: 5, label: 'uses' },
  { from: 1, to: 6, label: 'uses' },
  { from: 1, to: 7, label: 'uses' },
  { from: 1, to: 8, label: 'uses' },
  { from: 1, to: 9, label: 'uses' },
  { from: 1, to: 10, label: 'uses' },

  // Knowledge Engine connections
  { from: 1, to: 106, label: 'contains' },
  { from: 106, to: 1061, label: 'manages' },
  { from: 106, to: 1062, label: 'uses' },
  { from: 106, to: 1063, label: 'uses' },
  { from: 106, to: 1064, label: 'uses' },
  { from: 106, to: 1065, label: 'uses' },
  { from: 106, to: 1066, label: 'uses' },
  { from: 106, to: 1067, label: 'uses' },
  { from: 106, to: 1068, label: 'uses' },
  { from: 106, to: 1069, label: 'uses' },
  { from: 106, to: 1070, label: 'uses' },
  { from: 106, to: 1071, label: 'uses' },

  // Data flow within KnowledgeEngine
  { from: 1068, to: 1061, label: 'generates embeddings for' },
  { from: 1069, to: 1061, label: 'extracts triples for' },
  { from: 1061, to: 1062, label: 'indexes' },
  { from: 1061, to: 1063, label: 'indexes' },
  { from: 1061, to: 1064, label: 'indexes' },
  { from: 1062, to: 106, label: 'informs query' },
  { from: 1063, to: 106, label: 'informs query' },
  { from: 1064, to: 106, label: 'informs query' },
  { from: 106, to: 1070, label: 'updates' },
  { from: 106, to: 1071, label: 'updates' },

  // Connections to other modules
  { from: 106, to: 2, label: 'uses embedding' }, // KnowledgeEngine uses Embedding module
  { from: 106, to: 3, label: 'uses storage' },  // KnowledgeEngine uses Storage module
  { from: 106, to: 4, label: 'supports query' }, // KnowledgeEngine supports Query module
  { from: 106, to: 10, label: 'uses extraction' }, // KnowledgeEngine uses Extraction module

  // KnowledgeGraph and KnowledgeEngine using core data structures
  { from: 1, to: 201, label: 'uses' },
  { from: 1, to: 205, label: 'uses' },
  { from: 1, to: 206, label: 'uses' },
  { from: 1, to: 207, label: 'uses' },
  { from: 1, to: 208, label: 'uses' },
  { from: 1, to: 209, label: 'uses' },
  { from: 1, to: 210, label: 'uses' },
  { from: 1, to: 211, label: 'uses' },
  { from: 1, to: 212, label: 'uses' },
  { from: 1, to: 213, label: 'uses' },
  { from: 1, to: 214, label: 'uses' },
  { from: 1, to: 215, label: 'uses' },

  { from: 106, to: 201, label: 'uses' },
  { from: 106, to: 202, label: 'uses' },
  { from: 106, to: 203, label: 'uses' },
  { from: 106, to: 204, label: 'uses' },
  { from: 106, to: 205, label: 'uses' },
  { from: 106, to: 206, label: 'uses' },
  { from: 106, to: 207, label: 'uses' },
  { from: 106, to: 208, label: 'uses' },
  { from: 106, to: 209, label: 'uses' },
  { from: 106, to: 210, label: 'uses' },
  { from: 106, to: 211, label: 'uses' },
  { from: 106, to: 212, label: 'uses' },
  { from: 106, to: 213, label: 'uses' },
  { from: 106, to: 214, label: 'uses' },
  { from: 106, to: 215, label: 'uses' },

  // EntityStore manages EntityMeta and EntityData
  { from: 3, to: 205, label: 'manages' },
  { from: 3, to: 206, label: 'manages' },

  // CSRGraph uses Relationship
  { from: 4, to: 207, label: 'uses' },

  // QueryResult contains ContextEntity and Relationship
  { from: 209, to: 208, label: 'contains' },
  { from: 209, to: 207, label: 'contains' },

  // ContextEntity uses EntityKey
  { from: 208, to: 201, label: 'uses' },

  // Relationship uses EntityKey and RelationshipType
  { from: 207, to: 201, label: 'uses' },
  { from: 207, to: 203, label: 'uses' },
  { from: 207, to: 204, label: 'uses' },

  // CompactEntity uses EntityKey, EntityMeta
  { from: 210, to: 201, label: 'uses' },
  { from: 210, to: 205, label: 'uses' },

  // QueryParams uses Embedding
  { from: 213, to: 5, label: 'uses' },

const coreContainer = document.getElementById('core-network');
const coreData = { nodes: coreNodes, edges: coreEdges };
const coreOptions = {
  layout: {
    hierarchical: {
      direction: "UD",
      sortMethod: "directed"
    }
  },
  edges: {
    arrows: 'to'
  },
  groups: {
    core: { color: { background: '#2196f3', border: '#1976d2' }, font: { color: '#ffffff' } },
    'core-storage': { color: { background: '#1565c0', border: '#0d47a1' }, font: { color: '#ffffff' } },
    embedding: { color: { background: '#4caf50', border: '#388e3c' }, font: { color: '#ffffff' } },
    indexing: { color: { background: '#ffc107', border: '#ffa000' }, font: { color: '#000000' } },
    concurrency: { color: { background: '#f44336', border: '#c62828' }, font: { color: '#ffffff' } },
    metadata: { color: { background: '#9e9e9e', border: '#616161' }, font: { color: '#ffffff' } },
    'core-engine': { color: { background: '#8bc34a', border: '#689f38' }, font: { color: '#ffffff' } },
    'core-engine-storage': { color: { background: '#cddc39', border: '#afb42b' }, font: { color: '#000000' } },
    'core-engine-index': { color: { background: '#ffeb3b', border: '#fbc02d' }, font: { color: '#000000' } },
    'core-engine-metadata': { color: { background: '#9e9e9e', border: '#616161' }, font: { color: '#ffffff' } },
    'core-engine-embedding': { color: { background: '#4caf50', border: '#388e3c' }, font: { color: '#ffffff' } },
    'core-engine-extraction': { color: { background: '#00bcd4', border: '#0097a7' }, font: { color: '#ffffff' } },
    'core-engine-monitoring': { color: { background: '#ff5722', border: '#e64a19' }, font: { color: '#ffffff' } },
    'core-data': { color: { background: '#e0e0e0', border: '#9e9e9e' }, font: { color: '#000000' } },
  }
};
const coreNetwork = new vis.Network(coreContainer, coreData, coreOptions);

  // API Visualization
  const apiEndpoints = [
    {
      name: "knowledge_search",
      description: "Search the knowledge graph for entities and relationships relevant to a query. Returns structured knowledge that can be used to ground LLM responses and reduce hallucinations.",
      input_schema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Natural language query describing what knowledge to retrieve"
          },
          max_entities: {
            type: "integer",
            description: "Maximum number of entities to return (default: 20, max: 100)",
            minimum: 1,
            maximum: 100,
            default: 20
          },
          max_depth: {
            type: "integer",
            description: "Maximum relationship depth to explore (default: 2, max: 6)",
            minimum: 1,
            maximum: 6,
            default: 2
          }
        },
        required: ["query"]
      }
    },
    {
      name: "entity_lookup",
      description: "Look up specific entities by ID or natural language description. Returns detailed entity information including properties and relationships.",
      input_schema: {
        type: "object",
        properties: {
          entity_id: {
            type: "integer",
            description: "Specific entity ID to look up"
          },
          description: {
            type: "string",
            description: "Natural language description of the entity to find"
          }
        },
        oneOf: [
          { required: ["entity_id"] },
          { required: ["description"] }
        ]
      }
    },
    {
      name: "find_connections",
      description: "Find relationships and connections between entities. Useful for understanding how concepts are related.",
      input_schema: {
        type: "object",
        properties: {
          entity_a: {
            type: "string",
            description: "First entity (ID or description)"
          },
          entity_b: {
            type: "string",
            description: "Second entity (ID or description)"
          },
          max_path_length: {
            type: "integer",
            description: "Maximum path length to search (default: 4)",
            minimum: 1,
            maximum: 8,
            default: 4
          }
        },
        required: ["entity_a", "entity_b"]
      }
    },
    {
      name: "expand_concept",
      description: "Expand a concept by finding related entities and building a comprehensive knowledge subgraph. Useful for exploring topics in depth.",
      input_schema: {
        type: "object",
        properties: {
          concept: {
            type: "string",
            description: "The concept or topic to expand"
          },
          expansion_depth: {
            type: "integer",
            description: "How deeply to expand the concept (default: 3)",
            minimum: 1,
            maximum: 5,
            default: 3
          },
          max_entities: {
            type: "integer",
            description: "Maximum entities to include in expansion (default: 50)",
            minimum: 10,
            maximum: 200,
            default: 50
          }
        },
        required: ["concept"]
      }
    },
    {
      name: "graph_statistics",
      description: "Get statistical information about the knowledge graph including size, coverage, and performance metrics.",
      input_schema: {
        type: "object",
        properties: {},
        additionalProperties: false
      }
    }
  ];

  const apiContainer = document.getElementById('api-endpoints');

  apiEndpoints.forEach(api => {
    const apiDiv = document.createElement('div');
    apiDiv.classList.add('api-endpoint');

    const apiName = document.createElement('h3');
    apiName.textContent = api.name;
    apiDiv.appendChild(apiName);

    const apiDescription = document.createElement('p');
    apiDescription.textContent = api.description;
    apiDiv.appendChild(apiDescription);

    const apiSchema = document.createElement('pre');
    apiSchema.textContent = JSON.stringify(api.input_schema, null, 2);
    apiDiv.appendChild(apiSchema);

    // Add a simple form for testing
    const testForm = document.createElement('form');
    testForm.innerHTML = `
      <h4>Test this API:</h4>
      ${Object.entries(api.input_schema.properties || {}).map(([key, value]) => `
        <label for="${api.name}-${key}">${key} (${value.type}):</label>
        <input type="${value.type === 'integer' ? 'number' : 'text'}" id="${api.name}-${key}" name="${key}" ${value.default ? `placeholder="Default: ${value.default}"` : ''}>
        <br>
      `).join('')}
      <button type="submit">Test</button>
      <pre class="api-response"></pre>
    `;
    apiDiv.appendChild(testForm);

    testForm.addEventListener('submit', (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);
      const params = {};
      for (let [key, value] of formData.entries()) {
        params[key] = value;
      }
      console.log(`Testing ${api.name} with params:`, params);
      // Simulate API response
      const responseDiv = event.target.querySelector('.api-response');
      responseDiv.textContent = `Simulated response for ${api.name}:
` + JSON.stringify({ 
        status: "success", 
        data: "Simulated data based on input: " + JSON.stringify(params), 
        performance: { query_time_ms: Math.floor(Math.random() * 100) + 10, entities_processed: Math.floor(Math.random() * 1000) + 100, relationships_processed: Math.floor(Math.random() * 5000) + 500 }
      }, null, 2);
    });

    apiContainer.appendChild(apiDiv);
  });

  // Data Structures Visualization
  const dataStructures = [
    {
      name: "EntityKey",
      description: "A unique identifier for an entity within the knowledge graph.",
      fields: [
        { name: "id", type: "u64" },
        { name: "version", type: "u32" },
      ]
    },
    {
      name: "AttributeValue",
      description: "An enum representing various types of attribute values (String, Number, Boolean, Array, Object, Vector, Null).",
      fields: [
        { name: "String", type: "String" },
        { name: "Number", type: "f64" },
        { name: "Boolean", type: "bool" },
        { name: "Array", type: "Vec<AttributeValue>" },
        { name: "Object", type: "HashMap<String, AttributeValue>" },
        { name: "Vector", type: "Vec<f32>" },
        { name: "Null", type: "()" },
      ]
    },
    {
      name: "RelationshipType",
      description: "An enum defining the type of relationship (Directed, Undirected, Weighted).",
      fields: [
        { name: "Directed", type: "()" },
        { name: "Undirected", type: "()" },
        { name: "Weighted", type: "()" },
      ]
    },
    {
      name: "Weight",
      description: "A newtype struct representing the weight of a relationship, validated to be between 0.0 and 1.0.",
      fields: [
        { name: "value", type: "f32" },
      ]
    },
    {
      name: "EntityMeta",
      description: "Metadata associated with an entity, stored compactly.",
      fields: [
        { name: "type_id", type: "u16" },
        { name: "embedding_offset", type: "u32" },
        { name: "property_offset", type: "u32" },
        { name: "degree", type: "u16" },
        { name: "last_accessed", type: "std::time::Instant" },
      ]
    },
    {
      name: "EntityData",
      description: "The core data for an entity, including its properties and embedding.",
      fields: [
        { name: "type_id", type: "u16" },
        { name: "properties", type: "String" },
        { name: "embedding", type: "Vec<f32>" },
      ]
    },
    {
      name: "Relationship",
      description: "Represents a relationship between two entities.",
      fields: [
        { name: "from", type: "EntityKey" },
        { name: "to", type: "EntityKey" },
        { name: "rel_type", type: "u8" },
        { name: "weight", type: "f32" },
      ]
    },
    {
      name: "ContextEntity",
      description: "An entity with additional context for query results.",
      fields: [
        { name: "id", type: "EntityKey" },
        { name: "similarity", type: "f32" },
        { name: "neighbors", type: "Vec<EntityKey>" },
        { name: "properties", type: "String" },
      ]
    },
    {
      name: "QueryResult",
      description: "The result of a knowledge graph query.",
      fields: [
        { name: "entities", type: "Vec<ContextEntity>" },
        { name: "relationships", type: "Vec<Relationship>" },
        { name: "confidence", type: "f32" },
        { name: "query_time_ms", type: "u64" },
      ]
    },
    {
      name: "CompactEntity",
      description: "An ultra-fast optimized representation of an entity for maximum performance.",
      fields: [
        { name: "id", type: "u32" },
        { name: "type_id", type: "u16" },
        { name: "degree", type: "u16" },
        { name: "embedding_offset", type: "u32" },
        { name: "property_offset", type: "u32" },
      ]
    },
    {
      name: "NeighborSlice",
      description: "A zero-copy slice for efficient neighbor access.",
      fields: [
        { name: "data", type: "*const u32" },
        { name: "len", type: "u16" },
      ]
    },
    {
      name: "SIMDRelationship",
      description: "SIMD-friendly relationship representation for batched operations.",
      fields: [
        { name: "from", type: "[u32; 8]" },
        { name: "to", type: "[u32; 8]" },
        { name: "rel_type", type: "[u8; 8]" },
        { name: "weight", type: "[f32; 8]" },
        { name: "count", type: "u8" },
      ]
    },
    {
      name: "QueryParams",
      description: "Zero-copy query parameters for maximum performance.",
      fields: [
        { name: "embedding", type: "*const f32" },
        { name: "embedding_dim", type: "u16" },
        { name: "max_entities", type: "u16" },
        { name: "max_depth", type: "u8" },
        { name: "similarity_threshold", type: "f32" },
      ]
    },
    {
      name: "GraphQuery",
      description: "General graph query parameters.",
      fields: [
        { name: "query_text", type: "String" },
        { name: "query_type", type: "String" },
        { name: "max_results", type: "usize" },
      ]
    },
    {
      name: "TraversalParams",
      description: "Parameters for graph traversal operations.",
      fields: [
        { name: "max_depth", type: "usize" },
        { name: "max_paths", type: "usize" },
        { name: "include_bidirectional", type: "bool" },
        { name: "edge_weight_threshold", type: "Option<f32>" },
      ]
    },
  ];

  const dataStructuresContainer = document.getElementById('data-structures-content');

  dataStructures.forEach(ds => {
    const dsDiv = document.createElement('div');
    dsDiv.classList.add('data-structure');

    const dsName = document.createElement('h3');
    dsName.textContent = ds.name;
    dsDiv.appendChild(dsName);

    const dsDescription = document.createElement('p');
    dsDescription.textContent = ds.description;
    dsDiv.appendChild(dsDescription);

    const fieldsList = document.createElement('ul');
    ds.fields.forEach(field => {
      const fieldItem = document.createElement('li');
      fieldItem.textContent = `${field.name}: ${field.type}`;
      fieldsList.appendChild(fieldItem);
    });
    dsDiv.appendChild(fieldsList);

    dataStructuresContainer.appendChild(dsDiv);
  });

// Set the default tab
document.getElementsByClassName("tablinks")[0].click();
