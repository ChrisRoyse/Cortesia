import * as THREE from 'three';
import { KnowledgeGraphAnimator, GraphNode, GraphEdge } from './KnowledgeGraphAnimator';

export interface QueryStep {
    id: string;
    type: 'select' | 'filter' | 'join' | 'optional' | 'union' | 'graph';
    description: string;
    nodeIds: string[];
    edgeIds: string[];
    resultCount: number;
    duration: number;
    status: 'pending' | 'active' | 'completed' | 'failed';
}

export interface QueryPath {
    id: string;
    query: string;
    steps: QueryStep[];
    totalDuration: number;
    resultSet: string[];
    optimized: boolean;
}

export interface QueryVisualizationConfig {
    stepDuration: number;
    highlightColor: THREE.Color;
    pathColor: THREE.Color;
    resultColor: THREE.Color;
    animationSpeed: number;
    showIntermediateResults: boolean;
    debugMode: boolean;
}

export interface PerformanceMetrics {
    stepTimes: Map<string, number>;
    totalExecutionTime: number;
    nodesVisited: number;
    edgesTraversed: number;
    bottlenecks: string[];
}

export class QueryPathVisualizer {
    private graphAnimator: KnowledgeGraphAnimator;
    private config: QueryVisualizationConfig;
    private activeQuery: QueryPath | null = null;
    private currentStep: number = 0;
    private highlightedElements: Set<string> = new Set();
    private pathLines: THREE.Line[] = [];
    private resultMarkers: THREE.Mesh[] = [];
    private performanceMetrics: PerformanceMetrics;
    private stepAnimationId: number | null = null;

    constructor(
        graphAnimator: KnowledgeGraphAnimator,
        config?: Partial<QueryVisualizationConfig>
    ) {
        this.graphAnimator = graphAnimator;
        this.config = {
            stepDuration: 1000,
            highlightColor: new THREE.Color(0xffaa00),
            pathColor: new THREE.Color(0x00aaff),
            resultColor: new THREE.Color(0x00ff00),
            animationSpeed: 1.0,
            showIntermediateResults: true,
            debugMode: false,
            ...config
        };

        this.performanceMetrics = {
            stepTimes: new Map(),
            totalExecutionTime: 0,
            nodesVisited: 0,
            edgesTraversed: 0,
            bottlenecks: []
        };
    }

    public async visualizeQuery(queryPath: QueryPath): Promise<void> {
        this.activeQuery = queryPath;
        this.currentStep = 0;
        this.clearPreviousVisualization();
        this.resetMetrics();

        if (this.config.debugMode) {
            console.log('Starting query visualization:', queryPath);
        }

        await this.animateQueryExecution();
    }

    private async animateQueryExecution(): Promise<void> {
        if (!this.activeQuery) return;

        const startTime = performance.now();

        for (let i = 0; i < this.activeQuery.steps.length; i++) {
            this.currentStep = i;
            const step = this.activeQuery.steps[i];
            
            step.status = 'active';
            await this.animateStep(step);
            step.status = 'completed';

            if (this.config.showIntermediateResults) {
                this.showIntermediateResults(step);
            }

            // Brief pause between steps
            await this.delay(200);
        }

        this.performanceMetrics.totalExecutionTime = performance.now() - startTime;
        this.showFinalResults();
        this.analyzePerformance();
    }

    private async animateStep(step: QueryStep): Promise<void> {
        const stepStartTime = performance.now();

        // Highlight nodes involved in this step
        this.highlightNodes(step.nodeIds, this.config.highlightColor);
        
        // Highlight edges involved in this step
        this.highlightEdges(step.edgeIds, this.config.pathColor);

        // Create path visualization
        if (step.nodeIds.length > 1) {
            this.createPathVisualization(step.nodeIds);
        }

        // Animate traversal
        await this.animateTraversal(step);

        // Update metrics
        const stepDuration = performance.now() - stepStartTime;
        this.performanceMetrics.stepTimes.set(step.id, stepDuration);
        this.performanceMetrics.nodesVisited += step.nodeIds.length;
        this.performanceMetrics.edgesTraversed += step.edgeIds.length;

        if (this.config.debugMode) {
            console.log(`Step ${step.id} completed in ${stepDuration.toFixed(2)}ms`);
        }
    }

    private highlightNodes(nodeIds: string[], color: THREE.Color): void {
        nodeIds.forEach(nodeId => {
            const node = this.graphAnimator.getNode(nodeId);
            if (node && node.mesh) {
                const material = node.mesh.material as THREE.MeshLambertMaterial;
                material.emissive.copy(color).multiplyScalar(0.3);
                this.highlightedElements.add(nodeId);
                
                // Animate node activation
                this.graphAnimator.updateNodeActivation(nodeId, 1.0);
            }
        });
    }

    private highlightEdges(edgeIds: string[], color: THREE.Color): void {
        edgeIds.forEach(edgeId => {
            const edge = this.graphAnimator.getEdge(edgeId);
            if (edge && edge.line) {
                const material = edge.line.material as THREE.LineBasicMaterial;
                material.color.copy(color);
                material.opacity = 1.0;
                this.highlightedElements.add(edgeId);
            }
        });
    }

    private createPathVisualization(nodeIds: string[]): void {
        const positions: THREE.Vector3[] = [];
        
        nodeIds.forEach(nodeId => {
            const node = this.graphAnimator.getNode(nodeId);
            if (node) {
                positions.push(node.position.clone());
            }
        });

        if (positions.length < 2) return;

        // Create smooth curve through nodes
        const curve = new THREE.CatmullRomCurve3(positions);
        const points = curve.getPoints(50);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        const material = new THREE.LineBasicMaterial({
            color: this.config.pathColor,
            linewidth: 3,
            transparent: true,
            opacity: 0.8
        });

        const line = new THREE.Line(geometry, material);
        this.pathLines.push(line);
        
        // Add to scene through the graph animator's scene
        const scene = (this.graphAnimator as any).scene;
        scene.add(line);
    }

    private async animateTraversal(step: QueryStep): Promise<void> {
        return new Promise((resolve) => {
            const duration = this.config.stepDuration / this.config.animationSpeed;
            let progress = 0;
            const startTime = performance.now();

            const animate = () => {
                const elapsed = performance.now() - startTime;
                progress = Math.min(elapsed / duration, 1);

                // Create traveling light effect along path
                this.updateTravelingLight(step, progress);

                // Pulse effect on active nodes
                step.nodeIds.forEach(nodeId => {
                    const intensity = 0.5 + 0.5 * Math.sin(progress * Math.PI * 4);
                    this.graphAnimator.updateNodeActivation(nodeId, intensity);
                });

                if (progress < 1) {
                    this.stepAnimationId = requestAnimationFrame(animate);
                } else {
                    resolve();
                }
            };

            animate();
        });
    }

    private updateTravelingLight(step: QueryStep, progress: number): void {
        if (step.nodeIds.length < 2) return;

        const positions: THREE.Vector3[] = [];
        step.nodeIds.forEach(nodeId => {
            const node = this.graphAnimator.getNode(nodeId);
            if (node) positions.push(node.position.clone());
        });

        // Create traveling light particle
        const curve = new THREE.CatmullRomCurve3(positions);
        const lightPosition = curve.getPointAt(progress);
        
        // Create or update light particle
        const geometry = new THREE.SphereGeometry(0.2, 8, 8);
        const material = new THREE.MeshBasicMaterial({
            color: this.config.pathColor,
            transparent: true,
            opacity: 0.8
        });
        
        const light = new THREE.Mesh(geometry, material);
        light.position.copy(lightPosition);
        
        const scene = (this.graphAnimator as any).scene;
        scene.add(light);
        
        // Remove after animation
        setTimeout(() => scene.remove(light), 100);
    }

    private showIntermediateResults(step: QueryStep): void {
        step.nodeIds.forEach(nodeId => {
            const node = this.graphAnimator.getNode(nodeId);
            if (node) {
                // Create result marker
                const geometry = new THREE.RingGeometry(0.8, 1.2, 8);
                const material = new THREE.MeshBasicMaterial({
                    color: this.config.resultColor,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.6
                });
                
                const marker = new THREE.Mesh(geometry, material);
                marker.position.copy(node.position);
                marker.lookAt(0, 0, 0);
                
                this.resultMarkers.push(marker);
                
                const scene = (this.graphAnimator as any).scene;
                scene.add(marker);

                // Animate marker
                const animate = () => {
                    marker.rotation.z += 0.02;
                    requestAnimationFrame(animate);
                };
                animate();
            }
        });
    }

    private showFinalResults(): void {
        if (!this.activeQuery) return;

        this.activeQuery.resultSet.forEach(nodeId => {
            const node = this.graphAnimator.getNode(nodeId);
            if (node && node.mesh) {
                // Create final result highlight
                const geometry = new THREE.SphereGeometry(1.5, 16, 16);
                const material = new THREE.MeshBasicMaterial({
                    color: this.config.resultColor,
                    transparent: true,
                    opacity: 0.3,
                    wireframe: true
                });
                
                const highlight = new THREE.Mesh(geometry, material);
                highlight.position.copy(node.position);
                
                const scene = (this.graphAnimator as any).scene;
                scene.add(highlight);
                
                // Pulsing animation
                let scale = 1.0;
                const animate = () => {
                    scale += 0.05;
                    if (scale > 1.5) scale = 1.0;
                    highlight.scale.setScalar(scale);
                    requestAnimationFrame(animate);
                };
                animate();
            }
        });
    }

    public async visualizeQueryPlan(queryPath: QueryPath): Promise<void> {
        this.clearPreviousVisualization();

        // Create query plan tree visualization
        const planNodes = this.createQueryPlanNodes(queryPath.steps);
        const planLayout = this.calculatePlanLayout(planNodes);
        
        await this.animateQueryPlan(planLayout);
    }

    private createQueryPlanNodes(steps: QueryStep[]): any[] {
        return steps.map((step, index) => ({
            id: step.id,
            type: step.type,
            description: step.description,
            level: this.calculateStepLevel(step, steps),
            dependencies: this.findStepDependencies(step, steps),
            cost: step.duration,
            selectivity: step.resultCount
        }));
    }

    private calculateStepLevel(step: QueryStep, allSteps: QueryStep[]): number {
        // Simple heuristic: earlier steps are higher level
        return allSteps.indexOf(step);
    }

    private findStepDependencies(step: QueryStep, allSteps: QueryStep[]): string[] {
        // Find steps that this step depends on based on shared nodes/edges
        const dependencies: string[] = [];
        const stepIndex = allSteps.indexOf(step);
        
        for (let i = 0; i < stepIndex; i++) {
            const prevStep = allSteps[i];
            const hasSharedNodes = step.nodeIds.some(id => prevStep.nodeIds.includes(id));
            const hasSharedEdges = step.edgeIds.some(id => prevStep.edgeIds.includes(id));
            
            if (hasSharedNodes || hasSharedEdges) {
                dependencies.push(prevStep.id);
            }
        }
        
        return dependencies;
    }

    private calculatePlanLayout(planNodes: any[]): any {
        // Create hierarchical layout for query plan
        const layout = {
            nodes: new Map(),
            edges: []
        };

        planNodes.forEach((node, index) => {
            layout.nodes.set(node.id, {
                ...node,
                position: new THREE.Vector3(
                    (index - planNodes.length / 2) * 3,
                    node.level * -2,
                    5
                )
            });
        });

        // Create edges between dependent steps
        planNodes.forEach(node => {
            node.dependencies.forEach((depId: string) => {
                layout.edges.push({
                    from: depId,
                    to: node.id,
                    type: 'dependency'
                });
            });
        });

        return layout;
    }

    private async animateQueryPlan(layout: any): Promise<void> {
        // Create plan visualization nodes
        layout.nodes.forEach((node: any) => {
            const geometry = new THREE.BoxGeometry(2, 1, 0.5);
            const material = new THREE.MeshLambertMaterial({
                color: this.getStepTypeColor(node.type),
                transparent: true,
                opacity: 0.8
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.copy(node.position);
            
            const scene = (this.graphAnimator as any).scene;
            scene.add(mesh);
        });

        // Create dependency arrows
        layout.edges.forEach((edge: any) => {
            const fromNode = layout.nodes.get(edge.from);
            const toNode = layout.nodes.get(edge.to);
            
            if (fromNode && toNode) {
                const points = [fromNode.position.clone(), toNode.position.clone()];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({
                    color: 0x666666,
                    linewidth: 2
                });
                
                const line = new THREE.Line(geometry, material);
                const scene = (this.graphAnimator as any).scene;
                scene.add(line);
            }
        });
    }

    private getStepTypeColor(stepType: string): number {
        const colors: { [key: string]: number } = {
            select: 0x4CAF50,
            filter: 0xFF9800,
            join: 0x2196F3,
            optional: 0x9C27B0,
            union: 0xF44336,
            graph: 0x607D8B
        };
        
        return colors[stepType] || 0x888888;
    }

    private analyzePerformance(): void {
        if (!this.performanceMetrics) return;

        // Find performance bottlenecks
        const stepTimes = Array.from(this.performanceMetrics.stepTimes.entries());
        const sortedSteps = stepTimes.sort((a, b) => b[1] - a[1]);
        
        // Mark top 20% as bottlenecks
        const bottleneckCount = Math.max(1, Math.floor(sortedSteps.length * 0.2));
        this.performanceMetrics.bottlenecks = sortedSteps
            .slice(0, bottleneckCount)
            .map(([stepId]) => stepId);

        if (this.config.debugMode) {
            console.log('Performance Analysis:', this.performanceMetrics);
        }

        // Highlight bottleneck nodes
        this.highlightBottlenecks();
    }

    private highlightBottlenecks(): void {
        if (!this.activeQuery) return;

        this.performanceMetrics.bottlenecks.forEach(stepId => {
            const step = this.activeQuery!.steps.find(s => s.id === stepId);
            if (step) {
                step.nodeIds.forEach(nodeId => {
                    const node = this.graphAnimator.getNode(nodeId);
                    if (node && node.mesh) {
                        // Add red warning highlight for bottlenecks
                        const material = node.mesh.material as THREE.MeshLambertMaterial;
                        material.emissive.setHex(0x330000);
                    }
                });
            }
        });
    }

    private clearPreviousVisualization(): void {
        // Clear highlights
        this.highlightedElements.forEach(elementId => {
            const node = this.graphAnimator.getNode(elementId);
            const edge = this.graphAnimator.getEdge(elementId);
            
            if (node && node.mesh) {
                const material = node.mesh.material as THREE.MeshLambertMaterial;
                material.emissive.setHex(0x000000);
                this.graphAnimator.updateNodeActivation(elementId, 0);
            }
            
            if (edge && edge.line) {
                const material = edge.line.material as THREE.LineBasicMaterial;
                material.color.copy(edge.color);
                material.opacity = 0.6;
            }
        });
        
        this.highlightedElements.clear();

        // Remove path lines
        const scene = (this.graphAnimator as any).scene;
        this.pathLines.forEach(line => scene.remove(line));
        this.pathLines.length = 0;

        // Remove result markers
        this.resultMarkers.forEach(marker => scene.remove(marker));
        this.resultMarkers.length = 0;

        // Stop current step animation
        if (this.stepAnimationId) {
            cancelAnimationFrame(this.stepAnimationId);
            this.stepAnimationId = null;
        }
    }

    private resetMetrics(): void {
        this.performanceMetrics = {
            stepTimes: new Map(),
            totalExecutionTime: 0,
            nodesVisited: 0,
            edgesTraversed: 0,
            bottlenecks: []
        };
    }

    private delay(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    public pauseAnimation(): void {
        if (this.stepAnimationId) {
            cancelAnimationFrame(this.stepAnimationId);
            this.stepAnimationId = null;
        }
    }

    public resumeAnimation(): void {
        if (this.activeQuery && this.currentStep < this.activeQuery.steps.length) {
            this.animateQueryExecution();
        }
    }

    public skipToStep(stepIndex: number): void {
        if (this.activeQuery && stepIndex < this.activeQuery.steps.length) {
            this.currentStep = stepIndex;
            this.clearPreviousVisualization();
            this.animateStep(this.activeQuery.steps[stepIndex]);
        }
    }

    public getPerformanceMetrics(): PerformanceMetrics {
        return { ...this.performanceMetrics };
    }

    public setVisualizationSpeed(speed: number): void {
        this.config.animationSpeed = Math.max(0.1, Math.min(5.0, speed));
    }

    public toggleDebugMode(): void {
        this.config.debugMode = !this.config.debugMode;
    }

    public exportQueryPlan(): any {
        if (!this.activeQuery) return null;

        return {
            query: this.activeQuery.query,
            steps: this.activeQuery.steps.map(step => ({
                id: step.id,
                type: step.type,
                description: step.description,
                duration: step.duration,
                resultCount: step.resultCount
            })),
            metrics: this.getPerformanceMetrics()
        };
    }
}