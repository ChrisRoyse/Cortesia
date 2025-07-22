/**
 * Cognitive Patterns Integration Tests
 * Tests cognitive pattern visualization and recognition systems
 */

import { JSDOM } from 'jsdom';
import * as THREE from 'three';
import { CognitivePatternVisualizer } from '../../src/cognitive/CognitivePatternVisualizer';
import { PatternEffects } from '../../src/cognitive/PatternEffects';
import { PatternInteractions } from '../../src/cognitive/PatternInteractions';
import { AbstractThinking } from '../../src/cognitive/patterns/AbstractThinking';
import { SystemsThinking } from '../../src/cognitive/patterns/SystemsThinking';
import { LateralThinking } from '../../src/cognitive/patterns/LateralThinking';
import { CriticalThinking } from '../../src/cognitive/patterns/CriticalThinking';

// Setup DOM environment
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="test-container"></div></body></html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.WebGLRenderingContext = {} as any;

describe('Cognitive Patterns Integration Tests', () => {
  let container: HTMLElement;
  let patternVisualizer: CognitivePatternVisualizer;
  let patternEffects: PatternEffects;
  let patternInteractions: PatternInteractions;
  let abstractThinking: AbstractThinking;
  let systemsThinking: SystemsThinking;
  let lateralThinking: LateralThinking;
  let criticalThinking: CriticalThinking;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
    
    patternVisualizer = new CognitivePatternVisualizer({
      container,
      enablePatternRecognition: true,
      enableInhibitoryPatterns: true,
      enableHierarchicalProcessing: true,
      enableRealTimeAnalysis: true
    });

    patternEffects = new PatternEffects({
      container,
      enableGlowEffects: true,
      enablePulseAnimation: true,
      enableConnectionVisualization: true
    });

    patternInteractions = new PatternInteractions({
      container,
      enableInteractiveExploration: true,
      enablePatternComparison: true,
      enableCollaborativeAnalysis: true
    });

    // Initialize specific thinking patterns
    abstractThinking = new AbstractThinking({
      enableAbstractionLayers: true,
      enableConceptualMapping: true,
      visualizationMode: 'hierarchical'
    });

    systemsThinking = new SystemsThinking({
      enableSystemBoundaries: true,
      enableFeedbackLoops: true,
      enableEmergentProperties: true
    });

    lateralThinking = new LateralThinking({
      enableRandomConnections: true,
      enableProvocation: true,
      enableAlternativePerspectives: true
    });

    criticalThinking = new CriticalThinking({
      enableArgumentVisualization: true,
      enableEvidenceMapping: true,
      enableLogicalStructure: true
    });
  });

  afterEach(() => {
    patternVisualizer?.dispose();
    patternEffects?.dispose();
    patternInteractions?.dispose();
    abstractThinking?.dispose();
    systemsThinking?.dispose();
    lateralThinking?.dispose();
    criticalThinking?.dispose();
  });

  describe('Pattern Recognition and Classification', () => {
    test('should identify and classify cognitive patterns from data', async () => {
      const inputData = {
        entities: ['concept_A', 'concept_B', 'concept_C', 'concept_D'],
        relationships: [
          { from: 'concept_A', to: 'concept_B', type: 'causes', strength: 0.8 },
          { from: 'concept_B', to: 'concept_C', type: 'influences', strength: 0.6 },
          { from: 'concept_C', to: 'concept_D', type: 'leads_to', strength: 0.9 },
          { from: 'concept_D', to: 'concept_A', type: 'feedback', strength: 0.4 }
        ],
        context: {
          domain: 'problem-solving',
          complexity: 'medium',
          timeframe: 'sequential'
        }
      };

      const recognizedPatterns = await patternVisualizer.recognizePatterns(inputData);
      
      expect(recognizedPatterns).toBeDefined();
      expect(recognizedPatterns.length).toBeGreaterThan(0);
      
      // Should identify systems thinking pattern due to feedback loop
      const systemsPattern = recognizedPatterns.find(p => p.type === 'systems-thinking');
      expect(systemsPattern).toBeDefined();
      expect(systemsPattern.confidence).toBeGreaterThan(0.7);
      
      // Should identify causal reasoning pattern
      const causalPattern = recognizedPatterns.find(p => p.subtype === 'causal-reasoning');
      expect(causalPattern).toBeDefined();
    });

    test('should detect hierarchical thinking patterns', async () => {
      const hierarchicalData = {
        entities: ['goal', 'subgoal_1', 'subgoal_2', 'task_1a', 'task_1b', 'task_2a'],
        relationships: [
          { from: 'goal', to: 'subgoal_1', type: 'decomposes_to', strength: 1.0 },
          { from: 'goal', to: 'subgoal_2', type: 'decomposes_to', strength: 1.0 },
          { from: 'subgoal_1', to: 'task_1a', type: 'breaks_down_to', strength: 0.9 },
          { from: 'subgoal_1', to: 'task_1b', type: 'breaks_down_to', strength: 0.9 },
          { from: 'subgoal_2', to: 'task_2a', type: 'breaks_down_to', strength: 0.8 }
        ],
        structure: {
          levels: 3,
          hierarchyType: 'decomposition'
        }
      };

      const patterns = await patternVisualizer.recognizePatterns(hierarchicalData);
      const hierarchicalPattern = patterns.find(p => p.type === 'abstract-thinking' && p.subtype === 'hierarchical-decomposition');
      
      expect(hierarchicalPattern).toBeDefined();
      expect(hierarchicalPattern.properties.levels).toBe(3);
      expect(hierarchicalPattern.properties.decompositionStrategy).toBeDefined();
    });

    test('should identify lateral thinking patterns', async () => {
      const lateralData = {
        entities: ['problem', 'constraint_1', 'constraint_2', 'random_word', 'analogy_source', 'creative_solution'],
        relationships: [
          { from: 'problem', to: 'constraint_1', type: 'limited_by', strength: 0.8 },
          { from: 'problem', to: 'constraint_2', type: 'limited_by', strength: 0.7 },
          { from: 'random_word', to: 'creative_solution', type: 'inspires', strength: 0.6 },
          { from: 'analogy_source', to: 'creative_solution', type: 'analogous_to', strength: 0.9 },
          { from: 'creative_solution', to: 'problem', type: 'solves', strength: 0.95 }
        ],
        techniques: ['random_stimulation', 'analogy', 'constraint_removal'],
        unconventional: true
      };

      const patterns = await patternVisualizer.recognizePatterns(lateralData);
      const lateralPattern = patterns.find(p => p.type === 'lateral-thinking');
      
      expect(lateralPattern).toBeDefined();
      expect(lateralPattern.techniques).toContain('random_stimulation');
      expect(lateralPattern.techniques).toContain('analogy');
      expect(lateralPattern.unconventionalityScore).toBeGreaterThan(0.7);
    });
  });

  describe('Pattern Visualization', () => {
    test('should visualize abstract thinking patterns', async () => {
      const abstractPattern = {
        id: 'abstract-pattern-1',
        type: 'abstract-thinking',
        concepts: [
          { id: 'concrete_1', abstractionLevel: 0, position: { x: -5, y: -2, z: 0 } },
          { id: 'concrete_2', abstractionLevel: 0, position: { x: 5, y: -2, z: 0 } },
          { id: 'category_1', abstractionLevel: 1, position: { x: -2, y: 0, z: 0 } },
          { id: 'category_2', abstractionLevel: 1, position: { x: 2, y: 0, z: 0 } },
          { id: 'principle', abstractionLevel: 2, position: { x: 0, y: 2, z: 0 } }
        ],
        abstractions: [
          { from: 'concrete_1', to: 'category_1', type: 'instance_of' },
          { from: 'concrete_2', to: 'category_2', type: 'instance_of' },
          { from: 'category_1', to: 'principle', type: 'exemplifies' },
          { from: 'category_2', to: 'principle', type: 'exemplifies' }
        ]
      };

      await abstractThinking.visualizePattern(abstractPattern);
      
      const visualization = abstractThinking.getVisualization('abstract-pattern-1');
      expect(visualization).toBeDefined();
      expect(visualization.layers.length).toBe(3); // 0, 1, 2 abstraction levels
      expect(visualization.connections.length).toBe(4);
      
      // Check abstraction hierarchy is properly represented
      const principleLayer = visualization.layers.find(l => l.level === 2);
      expect(principleLayer.concepts.length).toBe(1);
      expect(principleLayer.concepts[0].id).toBe('principle');
    });

    test('should visualize systems thinking patterns', async () => {
      const systemsPattern = {
        id: 'systems-pattern-1',
        type: 'systems-thinking',
        components: [
          { id: 'input_A', type: 'input', position: { x: -8, y: 0, z: 0 } },
          { id: 'process_1', type: 'process', position: { x: -4, y: 0, z: 0 } },
          { id: 'process_2', type: 'process', position: { x: 0, y: 0, z: 0 } },
          { id: 'output_A', type: 'output', position: { x: 4, y: 0, z: 0 } },
          { id: 'feedback', type: 'feedback', position: { x: 0, y: -4, z: 0 } }
        ],
        flows: [
          { from: 'input_A', to: 'process_1', type: 'data_flow' },
          { from: 'process_1', to: 'process_2', type: 'data_flow' },
          { from: 'process_2', to: 'output_A', type: 'data_flow' },
          { from: 'process_2', to: 'feedback', type: 'feedback_flow' },
          { from: 'feedback', to: 'process_1', type: 'control_flow' }
        ],
        boundaries: [
          { id: 'system_boundary', components: ['process_1', 'process_2', 'feedback'] }
        ],
        emergentProperties: ['self-regulation', 'adaptation']
      };

      await systemsThinking.visualizePattern(systemsPattern);
      
      const visualization = systemsThinking.getVisualization('systems-pattern-1');
      expect(visualization).toBeDefined();
      expect(visualization.components.length).toBe(5);
      expect(visualization.feedbackLoops.length).toBe(1);
      expect(visualization.systemBoundaries.length).toBe(1);
      expect(visualization.emergentProperties.length).toBe(2);
    });

    test('should visualize critical thinking patterns', async () => {
      const criticalPattern = {
        id: 'critical-pattern-1',
        type: 'critical-thinking',
        claims: [
          { id: 'main_claim', text: 'AI will replace human jobs', position: { x: 0, y: 2, z: 0 } },
          { id: 'sub_claim_1', text: 'AI is advancing rapidly', position: { x: -3, y: 0, z: 0 } },
          { id: 'sub_claim_2', text: 'Many jobs are automatable', position: { x: 3, y: 0, z: 0 } }
        ],
        evidence: [
          { id: 'evidence_1', type: 'statistical', strength: 0.8, supports: 'sub_claim_1' },
          { id: 'evidence_2', type: 'case_study', strength: 0.6, supports: 'sub_claim_2' },
          { id: 'counter_evidence', type: 'expert_opinion', strength: 0.7, opposes: 'main_claim' }
        ],
        arguments: [
          { from: 'sub_claim_1', to: 'main_claim', type: 'supports', strength: 0.7 },
          { from: 'sub_claim_2', to: 'main_claim', type: 'supports', strength: 0.8 },
          { from: 'counter_evidence', to: 'main_claim', type: 'challenges', strength: 0.7 }
        ],
        logicalStructure: 'deductive'
      };

      await criticalThinking.visualizePattern(criticalPattern);
      
      const visualization = criticalThinking.getVisualization('critical-pattern-1');
      expect(visualization).toBeDefined();
      expect(visualization.argumentMap.claims.length).toBe(3);
      expect(visualization.argumentMap.evidence.length).toBe(3);
      expect(visualization.argumentMap.relationships.length).toBe(3);
      
      // Check logical structure analysis
      expect(visualization.logicalAnalysis.structure).toBe('deductive');
      expect(visualization.logicalAnalysis.validity).toBeDefined();
      expect(visualization.logicalAnalysis.soundness).toBeDefined();
    });
  });

  describe('Pattern Effects and Animation', () => {
    test('should create visual effects for pattern activation', async () => {
      const patternId = 'test-pattern-effects';
      
      await patternVisualizer.createPattern({
        id: patternId,
        type: 'convergent-thinking',
        nodes: ['idea_1', 'idea_2', 'idea_3', 'synthesis'],
        strength: 0.8
      });

      await patternEffects.activatePattern({
        patternId,
        effectType: 'convergence-glow',
        intensity: 0.9,
        duration: 2000,
        color: new THREE.Color(0x00ff88)
      });

      const effect = patternEffects.getActiveEffect(patternId);
      expect(effect).toBeDefined();
      expect(effect.isActive).toBe(true);
      expect(effect.intensity).toBe(0.9);
      expect(effect.type).toBe('convergence-glow');
    });

    test('should animate pattern transitions', async () => {
      const transitionConfig = {
        fromPattern: 'divergent-thinking-1',
        toPattern: 'convergent-thinking-1',
        transitionType: 'fade-morph',
        duration: 3000,
        easing: 'ease-in-out'
      };

      await patternEffects.animateTransition(transitionConfig);

      const transition = patternEffects.getActiveTransition(transitionConfig.fromPattern, transitionConfig.toPattern);
      expect(transition).toBeDefined();
      expect(transition.progress).toBeGreaterThanOrEqual(0);
      expect(transition.isActive).toBe(true);

      // Wait for transition to complete
      await new Promise(resolve => setTimeout(resolve, 3100));

      const completedTransition = patternEffects.getActiveTransition(transitionConfig.fromPattern, transitionConfig.toPattern);
      expect(completedTransition.isComplete).toBe(true);
      expect(completedTransition.progress).toBe(1.0);
    });

    test('should handle multiple simultaneous pattern effects', async () => {
      const patterns = [
        { id: 'pattern-1', type: 'abstract-thinking' },
        { id: 'pattern-2', type: 'systems-thinking' },
        { id: 'pattern-3', type: 'lateral-thinking' },
        { id: 'pattern-4', type: 'critical-thinking' }
      ];

      // Create patterns
      for (const pattern of patterns) {
        await patternVisualizer.createPattern({
          ...pattern,
          nodes: [`node_${pattern.id}_1`, `node_${pattern.id}_2`],
          strength: 0.7
        });
      }

      // Activate effects simultaneously
      const effectPromises = patterns.map(pattern => 
        patternEffects.activatePattern({
          patternId: pattern.id,
          effectType: 'pulse',
          intensity: 0.8,
          duration: 1500,
          color: new THREE.Color().setHSL(Math.random(), 0.8, 0.6)
        })
      );

      await Promise.all(effectPromises);

      const activeEffects = patternEffects.getActiveEffects();
      expect(activeEffects.length).toBe(4);
      expect(activeEffects.every(effect => effect.isActive)).toBe(true);
    });
  });

  describe('Pattern Interactions', () => {
    test('should enable interactive pattern exploration', async () => {
      const patternId = 'interactive-pattern-1';
      
      await patternVisualizer.createPattern({
        id: patternId,
        type: 'systems-thinking',
        nodes: ['component_1', 'component_2', 'component_3'],
        connections: [
          { from: 'component_1', to: 'component_2', type: 'influences' },
          { from: 'component_2', to: 'component_3', type: 'causes' },
          { from: 'component_3', to: 'component_1', type: 'feedback' }
        ]
      });

      await patternInteractions.enableInteraction({
        patternId,
        interactionTypes: ['hover', 'click', 'drag'],
        highlightConnected: true,
        showDetails: true
      });

      // Simulate user interaction
      const interactionResult = await patternInteractions.handleInteraction({
        type: 'hover',
        target: 'component_1',
        position: { x: 100, y: 200 }
      });

      expect(interactionResult).toBeDefined();
      expect(interactionResult.highlightedNodes).toContain('component_2'); // Connected node
      expect(interactionResult.details).toBeDefined();
      expect(interactionResult.details.nodeInfo).toBeDefined();
    });

    test('should support pattern comparison', async () => {
      const pattern1 = {
        id: 'comparison-pattern-1',
        type: 'convergent-thinking',
        nodes: ['idea_1', 'idea_2', 'synthesis'],
        strength: 0.8,
        efficiency: 0.7
      };

      const pattern2 = {
        id: 'comparison-pattern-2',
        type: 'convergent-thinking',
        nodes: ['concept_A', 'concept_B', 'integration'],
        strength: 0.6,
        efficiency: 0.9
      };

      await patternVisualizer.createPattern(pattern1);
      await patternVisualizer.createPattern(pattern2);

      const comparison = await patternInteractions.comparePatterns({
        patternIds: ['comparison-pattern-1', 'comparison-pattern-2'],
        comparisonDimensions: ['structure', 'efficiency', 'strength'],
        visualizationMode: 'side-by-side'
      });

      expect(comparison).toBeDefined();
      expect(comparison.similarities.length).toBeGreaterThan(0);
      expect(comparison.differences.length).toBeGreaterThan(0);
      expect(comparison.structuralSimilarity).toBeGreaterThan(0.5); // Same type
      expect(comparison.recommendations).toBeDefined();
    });

    test('should handle collaborative pattern analysis', async () => {
      const collaborationSession = await patternInteractions.startCollaborativeSession({
        sessionId: 'collab-session-1',
        participants: ['analyst_1', 'analyst_2', 'expert_1'],
        focusPatterns: ['collaborative-pattern-1'],
        permissions: {
          'analyst_1': ['view', 'annotate', 'modify'],
          'analyst_2': ['view', 'annotate'],
          'expert_1': ['view', 'annotate', 'validate']
        }
      });

      expect(collaborationSession).toBeDefined();
      expect(collaborationSession.id).toBe('collab-session-1');
      expect(collaborationSession.participants.length).toBe(3);

      // Simulate collaborative annotation
      await patternInteractions.addCollaborativeAnnotation({
        sessionId: 'collab-session-1',
        participantId: 'analyst_1',
        annotation: {
          type: 'insight',
          target: 'collaborative-pattern-1',
          content: 'This pattern shows strong emergence properties',
          position: { x: 0, y: 0, z: 0 }
        }
      });

      const annotations = await patternInteractions.getCollaborativeAnnotations('collab-session-1');
      expect(annotations.length).toBe(1);
      expect(annotations[0].participantId).toBe('analyst_1');
      expect(annotations[0].content).toContain('emergence properties');
    });
  });

  describe('Performance and Integration', () => {
    test('should maintain performance with complex pattern networks', async () => {
      const nodeCount = 500;
      const patternCount = 50;

      // Create large pattern network
      const nodes = Array.from({ length: nodeCount }, (_, i) => `node_${i}`);
      
      for (let i = 0; i < patternCount; i++) {
        const patternNodes = nodes.slice(i * 10, (i + 1) * 10);
        await patternVisualizer.createPattern({
          id: `performance-pattern-${i}`,
          type: ['abstract-thinking', 'systems-thinking', 'lateral-thinking', 'critical-thinking'][i % 4] as any,
          nodes: patternNodes,
          strength: Math.random()
        });
      }

      // Measure pattern recognition performance
      const recognitionStart = performance.now();
      const patterns = await patternVisualizer.getActivePatterns();
      const recognitionEnd = performance.now();
      
      expect(patterns.length).toBe(patternCount);
      expect(recognitionEnd - recognitionStart).toBeLessThan(500); // 500ms

      // Measure rendering performance
      const renderStart = performance.now();
      patternVisualizer.render();
      const renderEnd = performance.now();
      
      expect(renderEnd - renderStart).toBeLessThan(50); // 50ms for one frame
    });

    test('should handle memory efficiently with pattern lifecycle', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Create and dispose many patterns
      for (let i = 0; i < 100; i++) {
        const patternId = `memory-test-pattern-${i}`;
        
        await patternVisualizer.createPattern({
          id: patternId,
          type: 'abstract-thinking',
          nodes: Array.from({ length: 20 }, (_, j) => `node_${i}_${j}`),
          strength: Math.random()
        });

        // Add effects
        await patternEffects.activatePattern({
          patternId,
          effectType: 'glow',
          intensity: 0.5,
          duration: 100
        });

        // Clean up immediately
        await patternVisualizer.removePattern(patternId);
        await patternEffects.deactivatePattern(patternId);
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be minimal
      expect(memoryIncrease).toBeLessThan(100 * 1024 * 1024); // 100MB
    });

    test('should integrate with knowledge graph data', async () => {
      const knowledgeGraphData = {
        entities: [
          { id: 'entity_1', type: 'concept', properties: { category: 'science' } },
          { id: 'entity_2', type: 'concept', properties: { category: 'technology' } },
          { id: 'entity_3', type: 'concept', properties: { category: 'engineering' } },
          { id: 'entity_4', type: 'concept', properties: { category: 'mathematics' } }
        ],
        relationships: [
          { from: 'entity_1', to: 'entity_2', type: 'influences', strength: 0.8 },
          { from: 'entity_2', to: 'entity_3', type: 'enables', strength: 0.9 },
          { from: 'entity_3', to: 'entity_4', type: 'relies_on', strength: 0.7 },
          { from: 'entity_4', to: 'entity_1', type: 'supports', strength: 0.6 }
        ]
      };

      const integratedPatterns = await patternVisualizer.integrateKnowledgeGraph(knowledgeGraphData);
      
      expect(integratedPatterns.length).toBeGreaterThan(0);
      
      // Should recognize STEM systems thinking pattern
      const stemPattern = integratedPatterns.find(p => 
        p.type === 'systems-thinking' && 
        p.domain === 'STEM'
      );
      
      expect(stemPattern).toBeDefined();
      expect(stemPattern.entities.length).toBe(4);
      expect(stemPattern.systemBoundary).toBeDefined();
    });
  });
});