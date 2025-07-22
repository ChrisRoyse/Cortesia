// Knowledge Graph Query Animation System for LLMKG Phase 4
// Complete visualization suite for knowledge graph operations

export {
    KnowledgeGraphAnimator,
    type GraphNode,
    type GraphEdge,
    type AnimationState,
    type GraphConfig
} from './KnowledgeGraphAnimator';

export {
    QueryPathVisualizer,
    type QueryStep,
    type QueryPath,
    type QueryVisualizationConfig,
    type PerformanceMetrics
} from './QueryPathVisualizer';

export {
    EntityRelationshipFlow,
    type EntityLifecycleEvent,
    type RelationshipEvent,
    type FlowVisualizationConfig,
    type EntityState,
    type RelationshipState
} from './EntityRelationshipFlow';

export {
    TripleStoreVisualizer,
    type Triple,
    type TripleOperation,
    type TripleTransaction,
    type VisualizationConfig as TripleVisualizationConfig,
    type TripleVisual,
    type TransactionVisual
} from './TripleStoreVisualizer';

// Main integration class for complete knowledge graph visualization
import * as THREE from 'three';
import { KnowledgeGraphAnimator } from './KnowledgeGraphAnimator';
import { QueryPathVisualizer } from './QueryPathVisualizer';
import { EntityRelationshipFlow } from './EntityRelationshipFlow';
import { TripleStoreVisualizer } from './TripleStoreVisualizer';

export interface KnowledgeGraphVisualizationConfig {
    container: HTMLElement;
    enableQueryVisualization?: boolean;
    enableEntityFlow?: boolean;
    enableTripleStore?: boolean;
    graphConfig?: any;
    queryConfig?: any;
    flowConfig?: any;
    tripleConfig?: any;
}

export class KnowledgeGraphVisualization {
    private graphAnimator: KnowledgeGraphAnimator;
    private queryVisualizer?: QueryPathVisualizer;
    private entityFlow?: EntityRelationshipFlow;
    private tripleStore?: TripleStoreVisualizer;
    private config: KnowledgeGraphVisualizationConfig;

    constructor(config: KnowledgeGraphVisualizationConfig) {
        this.config = config;
        
        // Initialize main graph animator
        this.graphAnimator = new KnowledgeGraphAnimator(
            config.container,
            config.graphConfig
        );

        // Initialize optional components
        if (config.enableQueryVisualization !== false) {
            this.queryVisualizer = new QueryPathVisualizer(
                this.graphAnimator,
                config.queryConfig
            );
        }

        if (config.enableEntityFlow !== false) {
            this.entityFlow = new EntityRelationshipFlow(
                this.graphAnimator,
                config.flowConfig
            );
        }

        if (config.enableTripleStore !== false) {
            this.tripleStore = new TripleStoreVisualizer(
                this.graphAnimator,
                config.tripleConfig
            );
        }
    }

    // Main graph operations
    public addNode(nodeData: any) {
        return this.graphAnimator.addNode(nodeData);
    }

    public addEdge(edgeData: any) {
        return this.graphAnimator.addEdge(edgeData);
    }

    public removeNode(nodeId: string) {
        this.graphAnimator.removeNode(nodeId);
    }

    public removeEdge(edgeId: string) {
        this.graphAnimator.removeEdge(edgeId);
    }

    // Query visualization
    public async visualizeQuery(queryPath: any) {
        if (this.queryVisualizer) {
            return this.queryVisualizer.visualizeQuery(queryPath);
        }
        throw new Error('Query visualization not enabled');
    }

    public async visualizeQueryPlan(queryPath: any) {
        if (this.queryVisualizer) {
            return this.queryVisualizer.visualizeQueryPlan(queryPath);
        }
        throw new Error('Query visualization not enabled');
    }

    // Entity and relationship flow
    public addEntityEvent(event: any) {
        if (this.entityFlow) {
            this.entityFlow.addEntityEvent(event);
        }
    }

    public addRelationshipEvent(event: any) {
        if (this.entityFlow) {
            this.entityFlow.addRelationshipEvent(event);
        }
    }

    // Triple store operations
    public addTriple(triple: any) {
        if (this.tripleStore) {
            this.tripleStore.addTriple(triple);
        }
    }

    public updateTriple(triple: any) {
        if (this.tripleStore) {
            this.tripleStore.updateTriple(triple);
        }
    }

    public deleteTriple(tripleId: string) {
        if (this.tripleStore) {
            this.tripleStore.deleteTriple(tripleId);
        }
    }

    public executeTransaction(transaction: any) {
        if (this.tripleStore) {
            this.tripleStore.executeTransaction(transaction);
        }
    }

    // Animation controls
    public startAnimation() {
        this.graphAnimator.startAnimation();
    }

    public stopAnimation() {
        this.graphAnimator.stopAnimation();
    }

    // Performance and debugging
    public getPerformanceMetrics() {
        return {
            graph: {
                nodeCount: this.graphAnimator.getNode ? Object.keys(this.graphAnimator.getNode).length : 0,
                edgeCount: this.graphAnimator.getEdge ? Object.keys(this.graphAnimator.getEdge).length : 0
            },
            query: this.queryVisualizer?.getPerformanceMetrics?.(),
            flow: this.entityFlow?.exportFlowData?.(),
            triples: {
                count: this.tripleStore?.getTripleCount?.(),
                transactions: this.tripleStore?.getTransactionCount?.()
            }
        };
    }

    // Cleanup
    public clear() {
        this.graphAnimator.clear();
        this.entityFlow?.clear();
        this.tripleStore?.clear();
    }

    public dispose() {
        this.graphAnimator.dispose();
        this.queryVisualizer = undefined;
        this.entityFlow?.dispose();
        this.tripleStore?.dispose();
    }

    // Getters for direct component access
    public get graph() {
        return this.graphAnimator;
    }

    public get query() {
        return this.queryVisualizer;
    }

    public get flow() {
        return this.entityFlow;
    }

    public get triples() {
        return this.tripleStore;
    }
}

// Utility functions for LLMKG integration
export const LLMKGVisualizationUtils = {
    // Convert LLMKG entities to visualization nodes
    createNodeFromEntity(entity: any) {
        return {
            id: entity.id || entity.entity_id,
            type: entity.type || 'entity',
            position: new THREE.Vector3(
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20
            ),
            velocity: new THREE.Vector3(0, 0, 0),
            mass: 1.0,
            size: Math.max(0.5, Math.min(2.0, (entity.confidence || 1.0) * 1.5)),
            activation: entity.activation || 0.0,
            color: new THREE.Color(entity.color || 0x4a90e2),
            userData: entity
        };
    },

    // Convert LLMKG relationships to visualization edges
    createEdgeFromRelationship(relationship: any) {
        return {
            id: relationship.id || `${relationship.source}-${relationship.target}`,
            source: relationship.source,
            target: relationship.target,
            type: relationship.type || 'relationship',
            strength: relationship.strength || 1.0,
            weight: relationship.weight || 1.0,
            color: new THREE.Color(relationship.color || 0x666666),
            animated: relationship.animated || false
        };
    },

    // Convert LLMKG query to visualization path
    createQueryPathFromLLMKG(llmkgQuery: any) {
        return {
            id: llmkgQuery.id || `query_${Date.now()}`,
            query: llmkgQuery.sparql || llmkgQuery.query,
            steps: llmkgQuery.execution_plan?.map((step: any, index: number) => ({
                id: `step_${index}`,
                type: step.operation || 'select',
                description: step.description || `Step ${index + 1}`,
                nodeIds: step.nodes || [],
                edgeIds: step.edges || [],
                resultCount: step.result_count || 0,
                duration: step.duration || 0,
                status: 'pending'
            })) || [],
            totalDuration: llmkgQuery.total_duration || 0,
            resultSet: llmkgQuery.results || [],
            optimized: llmkgQuery.optimized || false
        };
    },

    // Convert LLMKG triple to visualization triple
    createTripleFromLLMKG(llmkgTriple: any) {
        return {
            id: llmkgTriple.id || `${llmkgTriple.subject}-${llmkgTriple.predicate}-${llmkgTriple.object}`,
            subject: llmkgTriple.subject,
            predicate: llmkgTriple.predicate,
            object: llmkgTriple.object,
            confidence: llmkgTriple.confidence || 1.0,
            timestamp: llmkgTriple.timestamp || Date.now(),
            metadata: llmkgTriple.metadata
        };
    }
};

// Default configurations for different use cases
export const DefaultConfigurations = {
    // Configuration for large knowledge graphs (1000+ nodes)
    largeGraph: {
        graphConfig: {
            nodeSize: { min: 0.3, max: 1.5 },
            edgeWidth: { min: 0.05, max: 0.3 },
            forceStrength: 80.0,
            damping: 0.9,
            layoutSteps: 5
        },
        queryConfig: {
            stepDuration: 800,
            animationSpeed: 1.5,
            debugMode: false
        },
        flowConfig: {
            timeScale: 1.2,
            flowSpeed: 1.5,
            particleCount: 50,
            animateCreation: false
        },
        tripleConfig: {
            tripleLayout: 'hierarchical',
            batchSize: 20,
            renderEffects: false
        }
    },

    // Configuration for detailed analysis (smaller graphs)
    detailed: {
        graphConfig: {
            nodeSize: { min: 0.8, max: 4.0 },
            edgeWidth: { min: 0.2, max: 1.0 },
            forceStrength: 120.0,
            damping: 0.8,
            layoutSteps: 15
        },
        queryConfig: {
            stepDuration: 1500,
            animationSpeed: 0.8,
            debugMode: true,
            showIntermediateResults: true
        },
        flowConfig: {
            timeScale: 0.8,
            flowSpeed: 1.0,
            particleCount: 150,
            animateCreation: true,
            showHistory: true
        },
        tripleConfig: {
            tripleLayout: 'triangular',
            batchSize: 5,
            renderEffects: true,
            showTransactions: true
        }
    },

    // Configuration for real-time monitoring
    realtime: {
        graphConfig: {
            nodeSize: { min: 0.4, max: 2.0 },
            edgeWidth: { min: 0.1, max: 0.6 },
            forceStrength: 100.0,
            damping: 0.85,
            layoutSteps: 8
        },
        queryConfig: {
            stepDuration: 500,
            animationSpeed: 2.0,
            debugMode: false
        },
        flowConfig: {
            timeScale: 1.5,
            flowSpeed: 2.5,
            particleCount: 80,
            entityFadeTime: 3000
        },
        tripleConfig: {
            tripleLayout: 'linear',
            batchSize: 15,
            renderEffects: true
        }
    }
};