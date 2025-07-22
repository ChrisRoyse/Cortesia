// Knowledge Graph Query Animation Demo
// Demonstrates all features of the LLMKG Phase 4 visualization system

import * as THREE from 'three';
import {
    KnowledgeGraphVisualization,
    LLMKGVisualizationUtils,
    DefaultConfigurations
} from './index';

// Demo data structures
interface DemoEntity {
    id: string;
    type: 'person' | 'place' | 'concept' | 'event';
    name: string;
    properties: { [key: string]: any };
}

interface DemoRelationship {
    id: string;
    source: string;
    target: string;
    type: string;
    strength: number;
}

interface DemoQuery {
    sparql: string;
    description: string;
    expectedResults: string[];
}

export class KnowledgeGraphDemo {
    private visualization: KnowledgeGraphVisualization;
    private container: HTMLElement;
    private entities: DemoEntity[] = [];
    private relationships: DemoRelationship[] = [];
    private queries: DemoQuery[] = [];

    constructor(container: HTMLElement) {
        this.container = container;
        
        // Initialize visualization with detailed configuration
        this.visualization = new KnowledgeGraphVisualization({
            container,
            enableQueryVisualization: true,
            enableEntityFlow: true,
            enableTripleStore: true,
            ...DefaultConfigurations.detailed
        });

        this.setupDemoData();
        this.setupEventHandlers();
    }

    private setupDemoData(): void {
        // Create sample entities
        this.entities = [
            {
                id: 'albert_einstein',
                type: 'person',
                name: 'Albert Einstein',
                properties: { profession: 'physicist', born: 1879, nationality: 'German' }
            },
            {
                id: 'theory_of_relativity',
                type: 'concept',
                name: 'Theory of Relativity',
                properties: { field: 'physics', year: 1915 }
            },
            {
                id: 'princeton',
                type: 'place',
                name: 'Princeton University',
                properties: { type: 'university', location: 'New Jersey' }
            },
            {
                id: 'nobel_prize',
                type: 'event',
                name: 'Nobel Prize in Physics 1921',
                properties: { year: 1921, category: 'physics' }
            },
            {
                id: 'spacetime',
                type: 'concept',
                name: 'Spacetime',
                properties: { field: 'physics', dimension: 4 }
            },
            {
                id: 'quantum_mechanics',
                type: 'concept',
                name: 'Quantum Mechanics',
                properties: { field: 'physics', probabilistic: true }
            }
        ];

        // Create sample relationships
        this.relationships = [
            {
                id: 'einstein_developed_relativity',
                source: 'albert_einstein',
                target: 'theory_of_relativity',
                type: 'developed',
                strength: 1.0
            },
            {
                id: 'einstein_worked_princeton',
                source: 'albert_einstein',
                target: 'princeton',
                type: 'worked_at',
                strength: 0.8
            },
            {
                id: 'einstein_won_nobel',
                source: 'albert_einstein',
                target: 'nobel_prize',
                type: 'received',
                strength: 0.9
            },
            {
                id: 'relativity_describes_spacetime',
                source: 'theory_of_relativity',
                target: 'spacetime',
                type: 'describes',
                strength: 0.95
            },
            {
                id: 'einstein_contributed_quantum',
                source: 'albert_einstein',
                target: 'quantum_mechanics',
                type: 'contributed_to',
                strength: 0.7
            }
        ];

        // Create sample queries
        this.queries = [
            {
                sparql: 'SELECT ?person ?concept WHERE { ?person developed ?concept }',
                description: 'Find people who developed concepts',
                expectedResults: ['albert_einstein', 'theory_of_relativity']
            },
            {
                sparql: 'SELECT ?entity WHERE { ?entity type person }',
                description: 'Find all people',
                expectedResults: ['albert_einstein']
            },
            {
                sparql: 'SELECT ?concept WHERE { albert_einstein ?relation ?concept }',
                description: 'Find everything related to Einstein',
                expectedResults: ['theory_of_relativity', 'princeton', 'nobel_prize', 'quantum_mechanics']
            }
        ];
    }

    private setupEventHandlers(): void {
        // Create control panel
        const controlPanel = this.createControlPanel();
        this.container.appendChild(controlPanel);

        // Setup keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            switch (event.key) {
                case '1':
                    this.runBasicVisualizationDemo();
                    break;
                case '2':
                    this.runQueryVisualizationDemo();
                    break;
                case '3':
                    this.runEntityFlowDemo();
                    break;
                case '4':
                    this.runTripleStoreDemo();
                    break;
                case '5':
                    this.runCompleteDemo();
                    break;
                case 'r':
                    this.reset();
                    break;
                case 'p':
                    this.pauseResume();
                    break;
            }
        });
    }

    private createControlPanel(): HTMLElement {
        const panel = document.createElement('div');
        panel.style.cssText = `
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 20px;
            border-radius: 8px;
            font-family: monospace;
            z-index: 1000;
            max-width: 300px;
        `;

        panel.innerHTML = `
            <h3>LLMKG Knowledge Graph Visualization Demo</h3>
            <div style="margin-bottom: 15px;">
                <button id="demo1" style="margin: 2px; padding: 8px;">1. Basic Graph</button>
                <button id="demo2" style="margin: 2px; padding: 8px;">2. Query Paths</button>
                <button id="demo3" style="margin: 2px; padding: 8px;">3. Entity Flow</button>
                <button id="demo4" style="margin: 2px; padding: 8px;">4. Triple Store</button>
                <button id="demo5" style="margin: 2px; padding: 8px;">5. Complete Demo</button>
            </div>
            <div style="margin-bottom: 15px;">
                <button id="reset" style="margin: 2px; padding: 8px;">Reset (R)</button>
                <button id="pause" style="margin: 2px; padding: 8px;">Pause (P)</button>
            </div>
            <div style="font-size: 12px; line-height: 1.4;">
                <strong>Features:</strong><br/>
                • 3D force-directed graph layout<br/>
                • Real-time query path visualization<br/>
                • Entity lifecycle animation<br/>
                • SPO triple store operations<br/>
                • Interactive graph exploration<br/>
                • Performance bottleneck analysis<br/>
                <br/>
                <strong>Controls:</strong><br/>
                • Mouse: hover/click nodes<br/>
                • Numbers: run demos<br/>
                • R: reset visualization<br/>
                • P: pause/resume animation
            </div>
        `;

        // Add event listeners
        panel.querySelector('#demo1')?.addEventListener('click', () => this.runBasicVisualizationDemo());
        panel.querySelector('#demo2')?.addEventListener('click', () => this.runQueryVisualizationDemo());
        panel.querySelector('#demo3')?.addEventListener('click', () => this.runEntityFlowDemo());
        panel.querySelector('#demo4')?.addEventListener('click', () => this.runTripleStoreDemo());
        panel.querySelector('#demo5')?.addEventListener('click', () => this.runCompleteDemo());
        panel.querySelector('#reset')?.addEventListener('click', () => this.reset());
        panel.querySelector('#pause')?.addEventListener('click', () => this.pauseResume());

        return panel;
    }

    public async runBasicVisualizationDemo(): Promise<void> {
        console.log('Running Basic Graph Visualization Demo...');
        this.reset();

        // Add entities as nodes
        this.entities.forEach(entity => {
            const node = LLMKGVisualizationUtils.createNodeFromEntity({
                ...entity,
                confidence: Math.random() * 0.5 + 0.5
            });
            this.visualization.addNode(node);
        });

        // Add relationships as edges with delay for animation
        for (const relationship of this.relationships) {
            await this.delay(500);
            const edge = LLMKGVisualizationUtils.createEdgeFromRelationship({
                ...relationship,
                animated: Math.random() > 0.5
            });
            this.visualization.addEdge(edge);
        }

        // Demonstrate activation propagation
        setTimeout(() => {
            console.log('Demonstrating activation propagation...');
            this.visualization.graph.animateActivationPropagation([
                'albert_einstein',
                'theory_of_relativity',
                'spacetime'
            ]);
        }, 3000);
    }

    public async runQueryVisualizationDemo(): Promise<void> {
        console.log('Running Query Visualization Demo...');
        await this.runBasicVisualizationDemo();

        // Wait for graph to settle
        await this.delay(2000);

        // Run each demo query
        for (const query of this.queries) {
            console.log(`Executing query: ${query.description}`);
            
            const queryPath = LLMKGVisualizationUtils.createQueryPathFromLLMKG({
                id: `demo_query_${Date.now()}`,
                sparql: query.sparql,
                execution_plan: [
                    {
                        operation: 'select',
                        description: 'Find matching entities',
                        nodes: this.entities.slice(0, 3).map(e => e.id),
                        edges: this.relationships.slice(0, 2).map(r => r.id),
                        result_count: query.expectedResults.length,
                        duration: Math.random() * 500 + 200
                    },
                    {
                        operation: 'filter',
                        description: 'Apply filters',
                        nodes: query.expectedResults,
                        edges: [],
                        result_count: query.expectedResults.length,
                        duration: Math.random() * 300 + 100
                    }
                ],
                results: query.expectedResults,
                optimized: true
            });

            if (this.visualization.query) {
                await this.visualization.visualizeQuery(queryPath);
                await this.delay(1000);
            }
        }

        // Show query plan visualization
        if (this.queries.length > 0 && this.visualization.query) {
            const complexQuery = LLMKGVisualizationUtils.createQueryPathFromLLMKG({
                id: 'complex_query_plan',
                sparql: 'Complex multi-step query demonstration',
                execution_plan: [
                    { operation: 'select', description: 'Initial selection', nodes: ['albert_einstein'], result_count: 1, duration: 150 },
                    { operation: 'join', description: 'Join with concepts', nodes: ['theory_of_relativity', 'quantum_mechanics'], result_count: 2, duration: 300 },
                    { operation: 'filter', description: 'Apply filters', nodes: [], result_count: 1, duration: 100 },
                    { operation: 'optional', description: 'Optional relations', nodes: ['princeton'], result_count: 1, duration: 200 }
                ]
            });
            
            await this.visualization.visualizeQueryPlan(complexQuery);
        }
    }

    public async runEntityFlowDemo(): Promise<void> {
        console.log('Running Entity Relationship Flow Demo...');
        this.reset();

        if (!this.visualization.flow) {
            console.warn('Entity flow not available');
            return;
        }

        // Demonstrate entity lifecycle events
        const entityEvents = [
            {
                id: 'create_einstein',
                timestamp: Date.now(),
                entityId: 'albert_einstein',
                type: 'create',
                data: { type: 'person', strength: 1.0, size: 1.5 }
            },
            {
                id: 'create_relativity',
                timestamp: Date.now() + 1000,
                entityId: 'theory_of_relativity',
                type: 'create',
                data: { type: 'concept', strength: 0.8, size: 1.2 }
            },
            {
                id: 'update_einstein',
                timestamp: Date.now() + 2000,
                entityId: 'albert_einstein',
                type: 'update',
                data: { strength: 1.2, recognition: 'increased' }
            },
            {
                id: 'create_quantum',
                timestamp: Date.now() + 3000,
                entityId: 'quantum_mechanics',
                type: 'create',
                data: { type: 'concept', strength: 0.9, size: 1.1 }
            }
        ];

        // Demonstrate relationship events
        const relationshipEvents = [
            {
                id: 'form_developed',
                timestamp: Date.now() + 1500,
                sourceEntity: 'albert_einstein',
                targetEntity: 'theory_of_relativity',
                relationType: 'developed',
                type: 'form',
                strength: 0.8,
                confidence: 0.9
            },
            {
                id: 'strengthen_developed',
                timestamp: Date.now() + 4000,
                sourceEntity: 'albert_einstein',
                targetEntity: 'theory_of_relativity',
                relationType: 'developed',
                type: 'strengthen',
                strength: 1.0,
                confidence: 0.95
            },
            {
                id: 'form_contributed',
                timestamp: Date.now() + 3500,
                sourceEntity: 'albert_einstein',
                targetEntity: 'quantum_mechanics',
                relationType: 'contributed_to',
                type: 'form',
                strength: 0.6,
                confidence: 0.7
            }
        ];

        // Add events to visualization
        entityEvents.forEach(event => {
            this.visualization.addEntityEvent(event);
        });

        relationshipEvents.forEach(event => {
            this.visualization.addRelationshipEvent(event);
        });

        // Add some basic graph nodes for context
        this.entities.slice(0, 4).forEach(entity => {
            const node = LLMKGVisualizationUtils.createNodeFromEntity(entity);
            this.visualization.addNode(node);
        });

        console.log('Entity flow events added - watch for lifecycle animations');
    }

    public async runTripleStoreDemo(): Promise<void> {
        console.log('Running Triple Store Visualization Demo...');
        this.reset();

        if (!this.visualization.triples) {
            console.warn('Triple store visualization not available');
            return;
        }

        // Create sample triples
        const triples = [
            {
                subject: 'albert_einstein',
                predicate: 'developed',
                object: 'theory_of_relativity',
                confidence: 0.95
            },
            {
                subject: 'albert_einstein',
                predicate: 'worked_at',
                object: 'princeton',
                confidence: 0.85
            },
            {
                subject: 'theory_of_relativity',
                predicate: 'describes',
                object: 'spacetime',
                confidence: 0.9
            },
            {
                subject: 'albert_einstein',
                predicate: 'received',
                object: 'nobel_prize',
                confidence: 1.0
            }
        ];

        // Add triples with delays to show insertion animation
        for (const tripleData of triples) {
            const triple = LLMKGVisualizationUtils.createTripleFromLLMKG(tripleData);
            this.visualization.addTriple(triple);
            await this.delay(800);
        }

        // Demonstrate atomic transaction
        await this.delay(2000);
        console.log('Demonstrating atomic transaction...');
        
        const transaction = {
            id: 'demo_transaction',
            operations: [
                {
                    id: 'insert_quantum_relation',
                    type: 'insert',
                    triple: LLMKGVisualizationUtils.createTripleFromLLMKG({
                        subject: 'albert_einstein',
                        predicate: 'contributed_to',
                        object: 'quantum_mechanics',
                        confidence: 0.75
                    }),
                    timestamp: Date.now()
                },
                {
                    id: 'update_relativity_confidence',
                    type: 'update',
                    triple: LLMKGVisualizationUtils.createTripleFromLLMKG({
                        id: triples[0].subject + '-' + triples[0].predicate + '-' + triples[0].object,
                        subject: triples[0].subject,
                        predicate: triples[0].predicate,
                        object: triples[0].object,
                        confidence: 0.99
                    }),
                    timestamp: Date.now() + 100
                }
            ],
            status: 'pending',
            timestamp: Date.now(),
            atomic: true
        };

        this.visualization.executeTransaction(transaction);

        // Demonstrate triple layout changes
        await this.delay(3000);
        console.log('Changing triple layout to hierarchical...');
        this.visualization.triples.setLayout('hierarchical');

        await this.delay(2000);
        console.log('Changing triple layout to circular...');
        this.visualization.triples.setLayout('circular');
    }

    public async runCompleteDemo(): Promise<void> {
        console.log('Running Complete Integrated Demo...');
        this.reset();

        // Phase 1: Build the knowledge graph
        console.log('Phase 1: Building knowledge graph...');
        await this.runBasicVisualizationDemo();

        // Phase 2: Add entity lifecycle events
        console.log('Phase 2: Adding entity lifecycle events...');
        await this.delay(2000);
        
        if (this.visualization.flow) {
            // Add some dynamic events
            this.visualization.addEntityEvent({
                id: 'strengthen_einstein',
                timestamp: Date.now(),
                entityId: 'albert_einstein',
                type: 'update',
                data: { strength: 1.5, importance: 'increased' }
            });

            this.visualization.addRelationshipEvent({
                id: 'strengthen_development',
                timestamp: Date.now() + 500,
                sourceEntity: 'albert_einstein',
                targetEntity: 'theory_of_relativity',
                relationType: 'developed',
                type: 'strengthen',
                strength: 1.0,
                confidence: 0.98
            });
        }

        // Phase 3: Execute queries
        console.log('Phase 3: Executing knowledge graph queries...');
        await this.delay(3000);
        
        if (this.visualization.query) {
            const complexQuery = LLMKGVisualizationUtils.createQueryPathFromLLMKG({
                id: 'integrated_demo_query',
                sparql: 'SELECT ?person ?achievement WHERE { ?person developed ?achievement . ?person received ?award }',
                execution_plan: [
                    {
                        operation: 'select',
                        description: 'Find all persons',
                        nodes: ['albert_einstein'],
                        edges: [],
                        result_count: 1,
                        duration: 200
                    },
                    {
                        operation: 'join',
                        description: 'Join with achievements',
                        nodes: ['albert_einstein', 'theory_of_relativity'],
                        edges: ['einstein_developed_relativity'],
                        result_count: 1,
                        duration: 350
                    },
                    {
                        operation: 'join',
                        description: 'Join with awards',
                        nodes: ['albert_einstein', 'theory_of_relativity', 'nobel_prize'],
                        edges: ['einstein_developed_relativity', 'einstein_won_nobel'],
                        result_count: 1,
                        duration: 300
                    },
                    {
                        operation: 'filter',
                        description: 'Apply final filters',
                        nodes: ['albert_einstein', 'theory_of_relativity', 'nobel_prize'],
                        edges: [],
                        result_count: 1,
                        duration: 150
                    }
                ],
                results: ['albert_einstein', 'theory_of_relativity', 'nobel_prize'],
                total_duration: 1000,
                optimized: true
            });

            await this.visualization.visualizeQuery(complexQuery);
        }

        // Phase 4: Triple store operations
        console.log('Phase 4: Triple store operations...');
        await this.delay(2000);
        
        if (this.visualization.triples) {
            // Add related triples
            const additionalTriples = [
                {
                    subject: 'spacetime',
                    predicate: 'property_of',
                    object: 'universe',
                    confidence: 0.9
                },
                {
                    subject: 'quantum_mechanics',
                    predicate: 'relates_to',
                    object: 'probability',
                    confidence: 0.85
                }
            ];

            for (const tripleData of additionalTriples) {
                const triple = LLMKGVisualizationUtils.createTripleFromLLMKG(tripleData);
                this.visualization.addTriple(triple);
                await this.delay(1000);
            }
        }

        // Phase 5: Performance analysis
        console.log('Phase 5: Performance analysis...');
        await this.delay(2000);
        
        const metrics = this.visualization.getPerformanceMetrics();
        console.log('Performance Metrics:', metrics);

        console.log('Complete integrated demo finished!');
    }

    private reset(): void {
        console.log('Resetting visualization...');
        this.visualization.clear();
    }

    private pauseResume(): void {
        // Toggle animation state
        try {
            this.visualization.stopAnimation();
            setTimeout(() => {
                this.visualization.startAnimation();
            }, 100);
            console.log('Animation toggled');
        } catch (error) {
            console.warn('Could not toggle animation:', error);
        }
    }

    private delay(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Export visualization data for analysis
    public exportData(): any {
        return {
            entities: this.entities,
            relationships: this.relationships,
            queries: this.queries,
            metrics: this.visualization.getPerformanceMetrics()
        };
    }

    // Clean up resources
    public dispose(): void {
        this.visualization.dispose();
    }
}

// Auto-start demo if running in browser
if (typeof window !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        const container = document.getElementById('knowledge-graph-container');
        if (container) {
            const demo = new KnowledgeGraphDemo(container);
            
            // Make demo globally available for debugging
            (window as any).knowledgeGraphDemo = demo;
            
            console.log('LLMKG Knowledge Graph Visualization Demo loaded!');
            console.log('Press 1-5 to run different demos, R to reset, P to pause');
        }
    });
}