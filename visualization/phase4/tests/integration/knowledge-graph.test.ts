/**
 * Knowledge Graph Integration Tests
 * Tests knowledge graph visualization, query animation, and entity relationship flow
 */

import { JSDOM } from 'jsdom';
import * as THREE from 'three';
import { KnowledgeGraphAnimator } from '../../src/knowledge/KnowledgeGraphAnimator';
import { QueryPathVisualizer } from '../../src/knowledge/QueryPathVisualizer';
import { EntityRelationshipFlow } from '../../src/knowledge/EntityRelationshipFlow';
import { TripleStoreVisualizer } from '../../src/knowledge/TripleStoreVisualizer';
import { KnowledgeGraphVisualization, DefaultConfigurations } from '../../src/knowledge';

// Setup DOM environment
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="test-container"></div></body></html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.WebGLRenderingContext = {} as any;

describe('Knowledge Graph Integration Tests', () => {
  let container: HTMLElement;
  let kgAnimator: KnowledgeGraphAnimator;
  let queryVisualizer: QueryPathVisualizer;
  let entityFlow: EntityRelationshipFlow;
  let tripleVisualizer: TripleStoreVisualizer;
  let kgVisualization: KnowledgeGraphVisualization;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
    
    kgAnimator = new KnowledgeGraphAnimator({
      container,
      enablePhysicsSimulation: true,
      enableInteractiveExploration: true,
      nodeSize: { min: 0.5, max: 3.0 },
      edgeWidth: { min: 0.1, max: 0.8 },
      forceStrength: 100.0,
      maxNodes: 1000
    });

    queryVisualizer = new QueryPathVisualizer({
      container,
      enableStepAnimation: true,
      enablePathHighlighting: true,
      stepDuration: 1000,
      animationSpeed: 1.0,
      showIntermediateResults: true
    });

    entityFlow = new EntityRelationshipFlow({
      container,
      enableLifecycleAnimation: true,
      enableRelationshipDynamics: true,
      flowSpeed: 2.0,
      particleCount: 100,
      enableHistory: true
    });

    tripleVisualizer = new TripleStoreVisualizer({
      container,
      layoutType: 'triangular',
      enableAtomicTransactions: true,
      enableBatchOperations: true,
      visualizeConfidence: true
    });

    kgVisualization = new KnowledgeGraphVisualization({
      container,
      enableQueryVisualization: true,
      enableEntityFlow: true,
      enableTripleStore: true,
      ...DefaultConfigurations.detailed
    });
  });

  afterEach(() => {
    kgAnimator?.dispose();
    queryVisualizer?.dispose();
    entityFlow?.dispose();
    tripleVisualizer?.dispose();
    kgVisualization?.dispose();
  });

  describe('Knowledge Graph Animation', () => {
    test('should create and animate 3D knowledge graph', async () => {
      const graphData = {
        nodes: [
          { id: 'person_1', type: 'Person', label: 'Alice', properties: { age: 30, city: 'New York' } },
          { id: 'person_2', type: 'Person', label: 'Bob', properties: { age: 25, city: 'San Francisco' } },
          { id: 'company_1', type: 'Company', label: 'Tech Corp', properties: { industry: 'Technology' } },
          { id: 'project_1', type: 'Project', label: 'AI Initiative', properties: { budget: 1000000 } }
        ],
        edges: [
          { id: 'rel_1', source: 'person_1', target: 'company_1', type: 'WORKS_FOR', properties: { since: 2020 } },
          { id: 'rel_2', source: 'person_2', target: 'company_1', type: 'WORKS_FOR', properties: { since: 2021 } },
          { id: 'rel_3', source: 'person_1', target: 'project_1', type: 'LEADS', properties: { responsibility: 'PM' } },
          { id: 'rel_4', source: 'person_2', target: 'project_1', type: 'CONTRIBUTES_TO', properties: { role: 'Developer' } }
        ]
      };

      await kgAnimator.loadGraph(graphData);
      
      const scene = kgAnimator.getScene();
      expect(scene).toBeInstanceOf(THREE.Scene);
      expect(scene.children.length).toBeGreaterThan(0);

      // Verify nodes were created
      const nodeCount = await kgAnimator.getNodeCount();
      expect(nodeCount).toBe(4);

      // Verify edges were created  
      const edgeCount = await kgAnimator.getEdgeCount();
      expect(edgeCount).toBe(4);

      // Test physics simulation
      const initialPositions = await kgAnimator.getNodePositions();
      
      // Run simulation for a few steps
      for (let i = 0; i < 10; i++) {
        kgAnimator.updatePhysics(1/60); // 60 FPS
      }

      const finalPositions = await kgAnimator.getNodePositions();
      
      // Positions should have changed due to physics
      const hasMovement = Object.keys(initialPositions).some(nodeId => 
        !initialPositions[nodeId].equals(finalPositions[nodeId])
      );
      expect(hasMovement).toBe(true);
    });

    test('should handle large graphs efficiently', async () => {
      const nodeCount = 500;
      const edgeCount = 1000;

      // Generate large graph data
      const nodes = Array.from({ length: nodeCount }, (_, i) => ({
        id: `node_${i}`,
        type: ['Person', 'Company', 'Project', 'Concept'][i % 4],
        label: `Node ${i}`,
        properties: { index: i }
      }));

      const edges = Array.from({ length: edgeCount }, (_, i) => ({
        id: `edge_${i}`,
        source: `node_${i % nodeCount}`,
        target: `node_${(i + Math.floor(Math.random() * 50) + 1) % nodeCount}`,
        type: ['RELATED_TO', 'CONNECTED_TO', 'PART_OF'][i % 3],
        properties: { weight: Math.random() }
      }));

      const loadStart = performance.now();
      await kgAnimator.loadGraph({ nodes, edges });
      const loadEnd = performance.now();

      const loadTime = loadEnd - loadStart;
      expect(loadTime).toBeLessThan(5000); // Should load within 5 seconds

      // Test rendering performance
      const renderStart = performance.now();
      kgAnimator.render();
      const renderEnd = performance.now();

      const renderTime = renderEnd - renderStart;
      expect(renderTime).toBeLessThan(50); // Should render within 50ms
    });

    test('should support interactive exploration', async () => {
      const graphData = {
        nodes: [
          { id: 'central', type: 'Hub', label: 'Central Node' },
          { id: 'satellite_1', type: 'Node', label: 'Satellite 1' },
          { id: 'satellite_2', type: 'Node', label: 'Satellite 2' },
          { id: 'satellite_3', type: 'Node', label: 'Satellite 3' }
        ],
        edges: [
          { id: 'e1', source: 'central', target: 'satellite_1', type: 'CONNECTS' },
          { id: 'e2', source: 'central', target: 'satellite_2', type: 'CONNECTS' },
          { id: 'e3', source: 'central', target: 'satellite_3', type: 'CONNECTS' }
        ]
      };

      await kgAnimator.loadGraph(graphData);

      // Test hover interaction
      const hoverResult = await kgAnimator.handleHover({
        nodeId: 'central',
        position: { x: 0, y: 0, z: 0 }
      });

      expect(hoverResult).toBeDefined();
      expect(hoverResult.highlightedNodes).toContain('satellite_1');
      expect(hoverResult.highlightedNodes).toContain('satellite_2');
      expect(hoverResult.highlightedNodes).toContain('satellite_3');

      // Test click interaction
      const clickResult = await kgAnimator.handleClick({
        nodeId: 'central',
        position: { x: 0, y: 0, z: 0 },
        button: 0
      });

      expect(clickResult).toBeDefined();
      expect(clickResult.selectedNode.id).toBe('central');
      expect(clickResult.connectedNodes.length).toBe(3);

      // Test drag interaction
      const dragResult = await kgAnimator.handleDrag({
        nodeId: 'satellite_1',
        startPosition: { x: 0, y: 0, z: 0 },
        endPosition: { x: 5, y: 5, z: 0 }
      });

      expect(dragResult).toBeDefined();
      expect(dragResult.newPosition).toEqual({ x: 5, y: 5, z: 0 });
    });
  });

  describe('Query Path Visualization', () => {
    test('should visualize SPARQL query execution', async () => {
      const queryPath = {
        id: 'sparql_query_1',
        query: `
          SELECT ?person ?name ?company WHERE {
            ?person a :Person ;
                    :name ?name ;
                    :worksFor ?company .
          }
        `,
        steps: [
          {
            id: 'step_1',
            type: 'pattern_match',
            description: 'Find all persons',
            pattern: '?person a :Person',
            nodeIds: ['person_1', 'person_2'],
            duration: 100,
            resultCount: 2
          },
          {
            id: 'step_2',
            type: 'property_fetch',
            description: 'Fetch names',
            pattern: '?person :name ?name',
            nodeIds: ['person_1', 'person_2'],
            duration: 50,
            resultCount: 2
          },
          {
            id: 'step_3',
            type: 'join',
            description: 'Join with companies',
            pattern: '?person :worksFor ?company',
            nodeIds: ['person_1', 'person_2', 'company_1'],
            duration: 75,
            resultCount: 2
          }
        ],
        resultSet: [
          { person: 'person_1', name: 'Alice', company: 'company_1' },
          { person: 'person_2', name: 'Bob', company: 'company_1' }
        ],
        totalDuration: 225,
        optimizations: ['index_usage', 'join_reordering']
      };

      const visualization = await queryVisualizer.visualizeQuery(queryPath);
      expect(visualization).toBeDefined();
      expect(visualization.id).toBe('sparql_query_1');
      expect(visualization.steps.length).toBe(3);
      expect(visualization.isActive).toBe(true);

      // Wait for animation to complete
      await new Promise(resolve => setTimeout(resolve, 3500)); // 3 steps × 1000ms + buffer

      const completedViz = await queryVisualizer.getVisualization('sparql_query_1');
      expect(completedViz.isComplete).toBe(true);
      expect(completedViz.executionMetrics.totalTime).toBe(225);
    });

    test('should identify and highlight query bottlenecks', async () => {
      const slowQueryPath = {
        id: 'slow_query_1',
        query: 'SELECT * FROM large_table WHERE complex_condition',
        steps: [
          {
            id: 'fast_step',
            type: 'index_scan',
            description: 'Fast index lookup',
            duration: 50,
            resultCount: 100
          },
          {
            id: 'slow_step',
            type: 'full_scan',
            description: 'Slow full table scan',
            duration: 2000, // Bottleneck
            resultCount: 1000000
          },
          {
            id: 'filter_step',
            type: 'filter',
            description: 'Apply filters',
            duration: 100,
            resultCount: 50
          }
        ],
        totalDuration: 2150
      };

      const bottleneckAnalysis = await queryVisualizer.analyzeBottlenecks(slowQueryPath);
      expect(bottleneckAnalysis).toBeDefined();
      expect(bottleneckAnalysis.bottlenecks.length).toBe(1);
      expect(bottleneckAnalysis.bottlenecks[0].stepId).toBe('slow_step');
      expect(bottleneckAnalysis.bottlenecks[0].severity).toBe('high');
      expect(bottleneckAnalysis.bottlenecks[0].impactPercent).toBeCloseTo(93, 0); // ~93% of total time

      const recommendations = bottleneckAnalysis.optimizationRecommendations;
      expect(recommendations.length).toBeGreaterThan(0);
      expect(recommendations[0].type).toBe('add_index');
      expect(recommendations[0].priority).toBe('high');
    });

    test('should visualize query plan trees', async () => {
      const queryPlan = {
        id: 'complex_query_plan',
        rootNode: {
          id: 'projection',
          type: 'Projection',
          columns: ['name', 'age', 'department'],
          cost: 1000,
          children: [
            {
              id: 'join',
              type: 'HashJoin',
              condition: 'employees.dept_id = departments.id',
              cost: 900,
              children: [
                {
                  id: 'scan_employees',
                  type: 'TableScan',
                  table: 'employees',
                  cost: 400,
                  estimatedRows: 10000
                },
                {
                  id: 'scan_departments',
                  type: 'IndexScan',
                  table: 'departments',
                  index: 'idx_dept_id',
                  cost: 100,
                  estimatedRows: 50
                }
              ]
            }
          ]
        }
      };

      const planVisualization = await queryVisualizer.visualizeQueryPlan(queryPlan);
      expect(planVisualization).toBeDefined();
      expect(planVisualization.treeStructure).toBeDefined();
      expect(planVisualization.treeStructure.depth).toBe(3);
      expect(planVisualization.treeStructure.nodeCount).toBe(4);
      
      // Check cost visualization
      expect(planVisualization.costVisualization.totalCost).toBe(1000);
      expect(planVisualization.costVisualization.bottleneckNodes).toContain('scan_employees');
      
      // Check execution flow
      expect(planVisualization.executionFlow.length).toBe(4);
      expect(planVisualization.executionFlow[0].nodeId).toBe('scan_employees'); // Should start with leaf nodes
    });
  });

  describe('Entity Relationship Flow', () => {
    test('should animate entity lifecycle events', async () => {
      const lifecycleEvents = [
        {
          id: 'create_person',
          type: 'entity_create',
          timestamp: Date.now(),
          entityId: 'new_person',
          entityType: 'Person',
          properties: { name: 'Charlie', age: 28 },
          position: { x: 0, y: 0, z: 0 }
        },
        {
          id: 'update_person',
          type: 'entity_update',
          timestamp: Date.now() + 1000,
          entityId: 'new_person',
          changes: { age: 29, city: 'Boston' },
          oldValues: { age: 28 }
        },
        {
          id: 'merge_persons',
          type: 'entity_merge',
          timestamp: Date.now() + 2000,
          primaryEntityId: 'new_person',
          mergedEntityId: 'duplicate_person',
          mergedProperties: { email: 'charlie@example.com' }
        },
        {
          id: 'delete_person',
          type: 'entity_delete',
          timestamp: Date.now() + 3000,
          entityId: 'new_person',
          reason: 'data_cleanup'
        }
      ];

      for (const event of lifecycleEvents) {
        await entityFlow.addLifecycleEvent(event);
      }

      const flowVisualization = await entityFlow.getFlowVisualization('new_person');
      expect(flowVisualization).toBeDefined();
      expect(flowVisualization.lifecycleStages.length).toBe(4);
      expect(flowVisualization.lifecycleStages[0].type).toBe('entity_create');
      expect(flowVisualization.lifecycleStages[3].type).toBe('entity_delete');

      // Check animation state
      const animationState = await entityFlow.getAnimationState();
      expect(animationState.activeAnimations.length).toBeGreaterThan(0);
      expect(animationState.queuedEvents.length).toBeGreaterThanOrEqual(0);
    });

    test('should visualize relationship dynamics', async () => {
      const relationshipEvents = [
        {
          id: 'form_friendship',
          type: 'relationship_form',
          timestamp: Date.now(),
          sourceEntityId: 'person_1',
          targetEntityId: 'person_2',
          relationshipType: 'FRIEND_OF',
          strength: 0.7,
          properties: { since: '2023-01-01' }
        },
        {
          id: 'strengthen_friendship',
          type: 'relationship_strengthen',
          timestamp: Date.now() + 1000,
          relationshipId: 'form_friendship',
          newStrength: 0.9,
          reason: 'increased_interaction'
        },
        {
          id: 'weaken_friendship',
          type: 'relationship_weaken',
          timestamp: Date.now() + 2000,
          relationshipId: 'form_friendship',
          newStrength: 0.4,
          reason: 'decreased_contact'
        },
        {
          id: 'dissolve_friendship',
          type: 'relationship_dissolve',
          timestamp: Date.now() + 3000,
          relationshipId: 'form_friendship',
          reason: 'conflict'
        }
      ];

      for (const event of relationshipEvents) {
        await entityFlow.addRelationshipEvent(event);
      }

      const relationshipFlow = await entityFlow.getRelationshipFlow('form_friendship');
      expect(relationshipFlow).toBeDefined();
      expect(relationshipFlow.evolutionStages.length).toBe(4);
      expect(relationshipFlow.strengthHistory.length).toBe(4);
      
      // Check strength progression
      expect(relationshipFlow.strengthHistory[0].strength).toBe(0.7);
      expect(relationshipFlow.strengthHistory[1].strength).toBe(0.9);
      expect(relationshipFlow.strengthHistory[2].strength).toBe(0.4);
      expect(relationshipFlow.strengthHistory[3].strength).toBe(0); // Dissolved
    });

    test('should handle particle flow effects', async () => {
      // Set up entities and relationships
      await entityFlow.addEntity({ id: 'hub', type: 'Hub', position: { x: 0, y: 0, z: 0 } });
      await entityFlow.addEntity({ id: 'node_1', type: 'Node', position: { x: 5, y: 0, z: 0 } });
      await entityFlow.addEntity({ id: 'node_2', type: 'Node', position: { x: -5, y: 0, z: 0 } });

      await entityFlow.addRelationship({
        id: 'flow_1',
        source: 'hub',
        target: 'node_1',
        type: 'DATA_FLOW',
        flowRate: 10, // particles per second
        bidirectional: false
      });

      await entityFlow.addRelationship({
        id: 'flow_2',
        source: 'node_2',
        target: 'hub',
        type: 'DATA_FLOW',
        flowRate: 15,
        bidirectional: false
      });

      // Start particle flow simulation
      await entityFlow.startFlowSimulation({
        duration: 5000, // 5 seconds
        particleSpeed: 2.0,
        particleLifetime: 3.0
      });

      // Check particle effects
      await new Promise(resolve => setTimeout(resolve, 1000)); // Let simulation run

      const flowMetrics = await entityFlow.getFlowMetrics();
      expect(flowMetrics).toBeDefined();
      expect(flowMetrics.activeParticles).toBeGreaterThan(0);
      expect(flowMetrics.totalParticlesGenerated).toBeGreaterThan(0);
      expect(flowMetrics.flowPaths.length).toBe(2);

      const particleStates = await entityFlow.getParticleStates();
      expect(particleStates.inTransit.length).toBeGreaterThan(0);
    });
  });

  describe('Triple Store Visualization', () => {
    test('should visualize RDF triples with different layouts', async () => {
      const triples = [
        {
          id: 'triple_1',
          subject: 'ex:Alice',
          predicate: 'ex:knows',
          object: 'ex:Bob',
          confidence: 0.9,
          source: 'manual_input'
        },
        {
          id: 'triple_2',
          subject: 'ex:Alice',
          predicate: 'ex:worksFor',
          object: 'ex:TechCorp',
          confidence: 1.0,
          source: 'verified'
        },
        {
          id: 'triple_3',
          subject: 'ex:Bob',
          predicate: 'ex:livesIn',
          object: 'ex:SanFrancisco',
          confidence: 0.8,
          source: 'inferred'
        }
      ];

      // Test triangular layout
      await tripleVisualizer.setLayout('triangular');
      await tripleVisualizer.visualizeTriples(triples);
      
      let layout = await tripleVisualizer.getLayout();
      expect(layout.type).toBe('triangular');
      expect(layout.triplePositions.length).toBe(3);

      // Test circular layout
      await tripleVisualizer.setLayout('circular');
      await tripleVisualizer.updateVisualization();

      layout = await tripleVisualizer.getLayout();
      expect(layout.type).toBe('circular');

      // Test hierarchical layout
      await tripleVisualizer.setLayout('hierarchical');
      await tripleVisualizer.updateVisualization();

      layout = await tripleVisualizer.getLayout();
      expect(layout.type).toBe('hierarchical');
      expect(layout.levels).toBeGreaterThan(0);
    });

    test('should handle atomic transactions', async () => {
      const transaction = {
        id: 'atomic_tx_1',
        operations: [
          {
            type: 'insert',
            triple: {
              subject: 'ex:NewPerson',
              predicate: 'ex:name',
              object: '"John Doe"',
              confidence: 1.0
            }
          },
          {
            type: 'insert',
            triple: {
              subject: 'ex:NewPerson',
              predicate: 'ex:age',
              object: '35',
              confidence: 1.0
            }
          },
          {
            type: 'delete',
            tripleId: 'obsolete_triple_1'
          },
          {
            type: 'update',
            tripleId: 'existing_triple_1',
            changes: { confidence: 0.95 }
          }
        ],
        atomic: true,
        timestamp: Date.now()
      };

      const txResult = await tripleVisualizer.executeTransaction(transaction);
      expect(txResult).toBeDefined();
      expect(txResult.success).toBe(true);
      expect(txResult.transactionId).toBe('atomic_tx_1');
      expect(txResult.operationsCompleted).toBe(4);

      // Verify transaction visualization
      const txVisualization = await tripleVisualizer.getTransactionVisualization('atomic_tx_1');
      expect(txVisualization).toBeDefined();
      expect(txVisualization.operationEffects.length).toBe(4);
      expect(txVisualization.atomicBoundary).toBeDefined();
    });

    test('should process batch operations efficiently', async () => {
      const batchSize = 100;
      const batchTriples = Array.from({ length: batchSize }, (_, i) => ({
        id: `batch_triple_${i}`,
        subject: `ex:Entity${i}`,
        predicate: 'ex:type',
        object: 'ex:BatchEntity',
        confidence: 0.8 + (Math.random() * 0.2),
        source: 'batch_import'
      }));

      const batchStart = performance.now();
      const batchResult = await tripleVisualizer.processBatch({
        id: 'batch_operation_1',
        triples: batchTriples,
        operation: 'insert',
        batchSize: 20, // Process in chunks of 20
        enableVisualization: true
      });
      const batchEnd = performance.now();

      expect(batchResult).toBeDefined();
      expect(batchResult.success).toBe(true);
      expect(batchResult.processedCount).toBe(batchSize);
      expect(batchEnd - batchStart).toBeLessThan(5000); // Should complete within 5 seconds

      // Check batch visualization
      const batchVisualization = await tripleVisualizer.getBatchVisualization('batch_operation_1');
      expect(batchVisualization).toBeDefined();
      expect(batchVisualization.chunks.length).toBe(5); // 100 / 20 = 5 chunks
      expect(batchVisualization.progressAnimation).toBeDefined();
    });

    test('should visualize confidence levels', async () => {
      const confidenceTriples = [
        {
          id: 'high_conf',
          subject: 'ex:Fact1',
          predicate: 'ex:isTrue',
          object: 'true',
          confidence: 0.95
        },
        {
          id: 'medium_conf',
          subject: 'ex:Fact2',
          predicate: 'ex:isTrue',
          object: 'true',
          confidence: 0.7
        },
        {
          id: 'low_conf',
          subject: 'ex:Fact3',
          predicate: 'ex:isTrue',
          object: 'true',
          confidence: 0.3
        },
        {
          id: 'uncertain',
          subject: 'ex:Fact4',
          predicate: 'ex:isTrue',
          object: 'unknown',
          confidence: 0.1
        }
      ];

      await tripleVisualizer.visualizeTriples(confidenceTriples);
      
      const confidenceVisualization = await tripleVisualizer.getConfidenceVisualization();
      expect(confidenceVisualization).toBeDefined();
      expect(confidenceVisualization.confidenceLevels.high).toBe(1); // high_conf
      expect(confidenceVisualization.confidenceLevels.medium).toBe(1); // medium_conf
      expect(confidenceVisualization.confidenceLevels.low).toBe(2); // low_conf + uncertain

      // Check visual mapping
      const visualMappings = confidenceVisualization.visualMappings;
      expect(visualMappings.opacity.max).toBeGreaterThan(visualMappings.opacity.min);
      expect(visualMappings.size.max).toBeGreaterThan(visualMappings.size.min);
      expect(visualMappings.colors.high).toBeDefined();
      expect(visualMappings.colors.low).toBeDefined();
    });
  });

  describe('Unified Knowledge Graph Visualization', () => {
    test('should integrate all visualization components', async () => {
      const integratedData = {
        entities: [
          { id: 'person_1', type: 'Person', label: 'Alice' },
          { id: 'company_1', type: 'Company', label: 'TechCorp' },
          { id: 'project_1', type: 'Project', label: 'AI Project' }
        ],
        relationships: [
          { id: 'rel_1', source: 'person_1', target: 'company_1', type: 'WORKS_FOR' },
          { id: 'rel_2', source: 'person_1', target: 'project_1', type: 'LEADS' }
        ],
        query: {
          id: 'integrated_query',
          sparql: 'SELECT ?person ?company WHERE { ?person :worksFor ?company }',
          steps: [
            { id: 'step_1', type: 'pattern_match', nodeIds: ['person_1'], duration: 100 },
            { id: 'step_2', type: 'join', nodeIds: ['person_1', 'company_1'], duration: 150 }
          ]
        }
      };

      // Load data into unified visualization
      await kgVisualization.loadIntegratedData(integratedData);

      // Verify all components are active
      const systemState = await kgVisualization.getSystemState();
      expect(systemState.knowledgeGraph.active).toBe(true);
      expect(systemState.queryVisualization.active).toBe(true);
      expect(systemState.entityFlow.active).toBe(true);
      expect(systemState.tripleStore.active).toBe(true);

      // Execute integrated query visualization
      await kgVisualization.executeQuery(integratedData.query);

      const queryResult = await kgVisualization.getQueryResult('integrated_query');
      expect(queryResult).toBeDefined();
      expect(queryResult.visualizationSynchronized).toBe(true);
      expect(queryResult.affectedComponents.length).toBeGreaterThan(1);
    });

    test('should maintain performance with complex integrated visualizations', async () => {
      const complexData = {
        entities: Array.from({ length: 200 }, (_, i) => ({
          id: `entity_${i}`,
          type: ['Person', 'Company', 'Project', 'Concept'][i % 4],
          label: `Entity ${i}`
        })),
        relationships: Array.from({ length: 400 }, (_, i) => ({
          id: `rel_${i}`,
          source: `entity_${i % 200}`,
          target: `entity_${(i + Math.floor(Math.random() * 50) + 1) % 200}`,
          type: ['RELATED_TO', 'PART_OF', 'USES'][i % 3]
        })),
        queries: Array.from({ length: 10 }, (_, i) => ({
          id: `query_${i}`,
          steps: Array.from({ length: 3 }, (_, j) => ({
            id: `step_${i}_${j}`,
            type: 'pattern_match',
            duration: 50 + Math.random() * 100
          }))
        }))
      };

      const loadStart = performance.now();
      await kgVisualization.loadIntegratedData(complexData);
      const loadEnd = performance.now();

      expect(loadEnd - loadStart).toBeLessThan(10000); // 10 seconds max

      // Test concurrent query execution
      const queryPromises = complexData.queries.map(query => 
        kgVisualization.executeQuery(query)
      );

      const queryStart = performance.now();
      await Promise.all(queryPromises);
      const queryEnd = performance.now();

      expect(queryEnd - queryStart).toBeLessThan(15000); // 15 seconds max

      // Test rendering performance
      const renderStart = performance.now();
      kgVisualization.render();
      const renderEnd = performance.now();

      expect(renderEnd - renderStart).toBeLessThan(100); // 100ms max per frame
    });

    test('should provide comprehensive analytics and insights', async () => {
      const analyticsData = {
        entities: [
          { id: 'e1', type: 'Person', centrality: 0.8 },
          { id: 'e2', type: 'Person', centrality: 0.6 },
          { id: 'e3', type: 'Company', centrality: 0.9 },
          { id: 'e4', type: 'Project', centrality: 0.4 }
        ],
        relationships: [
          { id: 'r1', source: 'e1', target: 'e3', weight: 0.9 },
          { id: 'r2', source: 'e2', target: 'e3', weight: 0.7 },
          { id: 'r3', source: 'e1', target: 'e4', weight: 0.8 }
        ],
        queries: [
          { id: 'q1', complexity: 'high', executionTime: 500, resultSize: 100 },
          { id: 'q2', complexity: 'medium', executionTime: 200, resultSize: 50 },
          { id: 'q3', complexity: 'low', executionTime: 100, resultSize: 10 }
        ]
      };

      await kgVisualization.loadIntegratedData(analyticsData);
      
      const analytics = await kgVisualization.generateAnalytics();
      expect(analytics).toBeDefined();

      // Graph structure analytics
      expect(analytics.graphStructure.nodeCount).toBe(4);
      expect(analytics.graphStructure.edgeCount).toBe(3);
      expect(analytics.graphStructure.density).toBeDefined();
      expect(analytics.graphStructure.clustering).toBeDefined();

      // Centrality analysis
      expect(analytics.centralityAnalysis.highCentralityNodes).toContain('e3');
      expect(analytics.centralityAnalysis.hubs.length).toBeGreaterThan(0);

      // Query performance analytics
      expect(analytics.queryPerformance.averageExecutionTime).toBeDefined();
      expect(analytics.queryPerformance.complexityDistribution).toBeDefined();
      expect(analytics.queryPerformance.bottlenecks).toBeDefined();

      // Recommendations
      expect(analytics.recommendations.length).toBeGreaterThan(0);
      expect(analytics.recommendations[0].type).toBeDefined();
      expect(analytics.recommendations[0].priority).toBeDefined();
    });
  });
});