import * as THREE from 'three';
import { KnowledgeGraphAnimator, GraphNode, GraphEdge } from './KnowledgeGraphAnimator';

export interface Triple {
    id: string;
    subject: string;
    predicate: string;
    object: string;
    confidence: number;
    timestamp: number;
    metadata?: any;
}

export interface TripleOperation {
    id: string;
    type: 'insert' | 'update' | 'delete' | 'query';
    triple: Triple;
    timestamp: number;
    transactionId?: string;
    batchId?: string;
}

export interface TripleTransaction {
    id: string;
    operations: TripleOperation[];
    status: 'pending' | 'committed' | 'rolled_back';
    timestamp: number;
    atomic: boolean;
}

export interface VisualizationConfig {
    tripleLayout: 'linear' | 'triangular' | 'circular' | 'hierarchical';
    animationDuration: number;
    showTransactions: boolean;
    highlightOperations: boolean;
    batchSize: number;
    renderEffects: boolean;
    spatialClustering: boolean;
}

export interface TripleVisual {
    triple: Triple;
    subjectNode: THREE.Mesh;
    predicateEdge: THREE.Line;
    objectNode: THREE.Mesh;
    predicateLabel: THREE.Sprite;
    connectionLine?: THREE.Line;
    atomicBounds?: THREE.Box3Helper;
}

export interface TransactionVisual {
    transaction: TripleTransaction;
    boundingBox: THREE.Box3Helper;
    operationMarkers: THREE.Mesh[];
    progressRing: THREE.Mesh;
    statusIndicator: THREE.Mesh;
}

export class TripleStoreVisualizer {
    private graphAnimator: KnowledgeGraphAnimator;
    private config: VisualizationConfig;
    private scene: THREE.Scene;
    private triples: Map<string, TripleVisual> = new Map();
    private transactions: Map<string, TransactionVisual> = new Map();
    private operationQueue: TripleOperation[] = [];
    private spatialIndex: Map<string, THREE.Vector3> = new Map();
    private animationMixer: THREE.AnimationMixer;
    private clock: THREE.Clock;
    private particleSystem: THREE.Points;
    private atomicEffects: THREE.Group = new THREE.Group();

    constructor(
        graphAnimator: KnowledgeGraphAnimator,
        config?: Partial<VisualizationConfig>
    ) {
        this.graphAnimator = graphAnimator;
        this.config = {
            tripleLayout: 'triangular',
            animationDuration: 1000,
            showTransactions: true,
            highlightOperations: true,
            batchSize: 10,
            renderEffects: true,
            spatialClustering: true,
            ...config
        };

        this.scene = (this.graphAnimator as any).scene;
        this.clock = new THREE.Clock();
        this.animationMixer = new THREE.AnimationMixer(this.scene);
        
        this.initializeParticleSystem();
        this.initializeAtomicEffects();
        this.startAnimationLoop();
    }

    private initializeParticleSystem(): void {
        const particleCount = 2000;
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);

        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 100;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 100;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 100;

            colors[i * 3] = Math.random() * 0.3 + 0.1;
            colors[i * 3 + 1] = Math.random() * 0.5 + 0.2;
            colors[i * 3 + 2] = Math.random() * 0.7 + 0.3;

            sizes[i] = Math.random() * 0.05 + 0.01;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        const material = new THREE.PointsMaterial({
            size: 0.02,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true,
            opacity: 0.4
        });

        this.particleSystem = new THREE.Points(geometry, material);
        this.scene.add(this.particleSystem);
    }

    private initializeAtomicEffects(): void {
        this.scene.add(this.atomicEffects);
    }

    public addTriple(triple: Triple): void {
        const operation: TripleOperation = {
            id: `insert_${triple.id}`,
            type: 'insert',
            triple,
            timestamp: performance.now()
        };
        
        this.operationQueue.push(operation);
        this.processOperations();
    }

    public updateTriple(triple: Triple): void {
        const operation: TripleOperation = {
            id: `update_${triple.id}`,
            type: 'update',
            triple,
            timestamp: performance.now()
        };
        
        this.operationQueue.push(operation);
        this.processOperations();
    }

    public deleteTriple(tripleId: string): void {
        const existingTriple = this.triples.get(tripleId);
        if (!existingTriple) return;

        const operation: TripleOperation = {
            id: `delete_${tripleId}`,
            type: 'delete',
            triple: existingTriple.triple,
            timestamp: performance.now()
        };
        
        this.operationQueue.push(operation);
        this.processOperations();
    }

    public executeTransaction(transaction: TripleTransaction): void {
        transaction.operations.forEach(op => {
            op.transactionId = transaction.id;
            this.operationQueue.push(op);
        });

        if (this.config.showTransactions) {
            this.visualizeTransaction(transaction);
        }

        this.processOperations();
    }

    private processOperations(): void {
        const batchSize = Math.min(this.config.batchSize, this.operationQueue.length);
        const batch = this.operationQueue.splice(0, batchSize);

        batch.forEach(operation => {
            switch (operation.type) {
                case 'insert':
                    this.insertTriple(operation);
                    break;
                case 'update':
                    this.updateTripleVisual(operation);
                    break;
                case 'delete':
                    this.deleteTripleVisual(operation);
                    break;
                case 'query':
                    this.queryTriples(operation);
                    break;
            }
        });

        // Continue processing if queue has more operations
        if (this.operationQueue.length > 0) {
            setTimeout(() => this.processOperations(), 50);
        }
    }

    private insertTriple(operation: TripleOperation): void {
        const triple = operation.triple;
        const tripleVisual = this.createTripleVisual(triple);
        
        this.triples.set(triple.id, tripleVisual);
        
        // Add to scene with animation
        if (this.config.highlightOperations) {
            this.animateTripleInsertion(tripleVisual);
        }
        
        // Update spatial index
        if (this.config.spatialClustering) {
            this.updateSpatialIndex(triple);
        }
        
        // Add atomic effect if part of transaction
        if (operation.transactionId && this.config.renderEffects) {
            this.addAtomicEffect(tripleVisual, 'insert');
        }
    }

    private createTripleVisual(triple: Triple): TripleVisual {
        const layout = this.calculateTripleLayout(triple);
        
        // Create subject node
        const subjectGeometry = new THREE.SphereGeometry(0.3, 16, 16);
        const subjectMaterial = new THREE.MeshLambertMaterial({
            color: this.getSubjectColor(triple.subject),
            transparent: true,
            opacity: 0.8
        });
        const subjectNode = new THREE.Mesh(subjectGeometry, subjectMaterial);
        subjectNode.position.copy(layout.subjectPosition);
        subjectNode.userData = { type: 'subject', tripleId: triple.id, value: triple.subject };
        this.scene.add(subjectNode);

        // Create object node
        const objectGeometry = new THREE.SphereGeometry(0.3, 16, 16);
        const objectMaterial = new THREE.MeshLambertMaterial({
            color: this.getObjectColor(triple.object),
            transparent: true,
            opacity: 0.8
        });
        const objectNode = new THREE.Mesh(objectGeometry, objectMaterial);
        objectNode.position.copy(layout.objectPosition);
        objectNode.userData = { type: 'object', tripleId: triple.id, value: triple.object };
        this.scene.add(objectNode);

        // Create predicate edge
        const edgePoints = [layout.subjectPosition, layout.objectPosition];
        const edgeGeometry = new THREE.BufferGeometry().setFromPoints(edgePoints);
        const edgeMaterial = new THREE.LineBasicMaterial({
            color: this.getPredicateColor(triple.predicate),
            linewidth: 2,
            transparent: true,
            opacity: 0.7
        });
        const predicateEdge = new THREE.Line(edgeGeometry, edgeMaterial);
        predicateEdge.userData = { type: 'predicate', tripleId: triple.id, value: triple.predicate };
        this.scene.add(predicateEdge);

        // Create predicate label
        const predicateLabel = this.createPredicateLabel(triple.predicate, layout.predicatePosition);
        this.scene.add(predicateLabel);

        // Create connection line for triangular layout
        let connectionLine;
        if (this.config.tripleLayout === 'triangular' || this.config.tripleLayout === 'circular') {
            const connectionPoints = [layout.subjectPosition, layout.predicatePosition, layout.objectPosition];
            const connectionGeometry = new THREE.BufferGeometry().setFromPoints(connectionPoints);
            const connectionMaterial = new THREE.LineBasicMaterial({
                color: 0x444444,
                transparent: true,
                opacity: 0.3
            });
            connectionLine = new THREE.Line(connectionGeometry, connectionMaterial);
            this.scene.add(connectionLine);
        }

        return {
            triple,
            subjectNode,
            predicateEdge,
            objectNode,
            predicateLabel,
            connectionLine
        };
    }

    private calculateTripleLayout(triple: Triple): any {
        const basePosition = this.getSpatialPosition(triple);
        
        switch (this.config.tripleLayout) {
            case 'linear':
                return {
                    subjectPosition: basePosition.clone().add(new THREE.Vector3(-2, 0, 0)),
                    predicatePosition: basePosition.clone(),
                    objectPosition: basePosition.clone().add(new THREE.Vector3(2, 0, 0))
                };
                
            case 'triangular':
                return {
                    subjectPosition: basePosition.clone().add(new THREE.Vector3(-1, -1, 0)),
                    predicatePosition: basePosition.clone().add(new THREE.Vector3(0, 1, 0)),
                    objectPosition: basePosition.clone().add(new THREE.Vector3(1, -1, 0))
                };
                
            case 'circular':
                const radius = 1.5;
                return {
                    subjectPosition: basePosition.clone().add(new THREE.Vector3(
                        Math.cos(0) * radius,
                        Math.sin(0) * radius,
                        0
                    )),
                    predicatePosition: basePosition.clone().add(new THREE.Vector3(
                        Math.cos(Math.PI * 2 / 3) * radius,
                        Math.sin(Math.PI * 2 / 3) * radius,
                        0
                    )),
                    objectPosition: basePosition.clone().add(new THREE.Vector3(
                        Math.cos(Math.PI * 4 / 3) * radius,
                        Math.sin(Math.PI * 4 / 3) * radius,
                        0
                    ))
                };
                
            case 'hierarchical':
                const level = this.calculateHierarchicalLevel(triple);
                return {
                    subjectPosition: basePosition.clone().add(new THREE.Vector3(-1.5, level * 2, 0)),
                    predicatePosition: basePosition.clone().add(new THREE.Vector3(0, level * 2 + 0.5, 0)),
                    objectPosition: basePosition.clone().add(new THREE.Vector3(1.5, level * 2, 0))
                };
                
            default:
                return this.calculateTripleLayout({ ...triple }); // Default to triangular
        }
    }

    private getSpatialPosition(triple: Triple): THREE.Vector3 {
        if (!this.config.spatialClustering) {
            return new THREE.Vector3(
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20
            );
        }

        // Use consistent hash-based positioning for related triples
        const hash = this.hashTriple(triple);
        const x = ((hash & 0xFF) / 255 - 0.5) * 30;
        const y = (((hash >> 8) & 0xFF) / 255 - 0.5) * 30;
        const z = (((hash >> 16) & 0xFF) / 255 - 0.5) * 30;
        
        return new THREE.Vector3(x, y, z);
    }

    private hashTriple(triple: Triple): number {
        // Simple hash function for consistent positioning
        let hash = 0;
        const str = `${triple.subject}-${triple.predicate}-${triple.object}`;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash);
    }

    private calculateHierarchicalLevel(triple: Triple): number {
        // Simple heuristic: count depth based on subject complexity
        return triple.subject.split('/').length - 1;
    }

    private createPredicateLabel(predicate: string, position: THREE.Vector3): THREE.Sprite {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d')!;
        canvas.width = 256;
        canvas.height = 64;
        
        context.fillStyle = 'rgba(0, 0, 0, 0.7)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        context.fillStyle = 'white';
        context.font = '16px Arial';
        context.textAlign = 'center';
        context.fillText(predicate, canvas.width / 2, canvas.height / 2 + 5);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ 
            map: texture,
            transparent: true,
            opacity: 0.8
        });
        
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position);
        sprite.scale.set(2, 0.5, 1);
        
        return sprite;
    }

    private getSubjectColor(subject: string): number {
        // Color based on subject type/namespace
        if (subject.includes('person')) return 0x4CAF50;
        if (subject.includes('place')) return 0x2196F3;
        if (subject.includes('thing')) return 0xFF9800;
        if (subject.includes('concept')) return 0x9C27B0;
        return 0x4a90e2;
    }

    private getPredicateColor(predicate: string): number {
        // Color based on predicate type
        if (predicate.includes('is') || predicate.includes('type')) return 0xFF5722;
        if (predicate.includes('has') || predicate.includes('owns')) return 0x4CAF50;
        if (predicate.includes('knows') || predicate.includes('related')) return 0x2196F3;
        if (predicate.includes('located') || predicate.includes('at')) return 0xFF9800;
        return 0x666666;
    }

    private getObjectColor(object: string): number {
        // Similar to subject coloring but slightly different hues
        if (object.includes('person')) return 0x66BB6A;
        if (object.includes('place')) return 0x42A5F5;
        if (object.includes('thing')) return 0xFFB74D;
        if (object.includes('concept')) return 0xAB47BC;
        return 0x5BA0F2;
    }

    private animateTripleInsertion(tripleVisual: TripleVisual): void {
        // Scale-in animation
        [tripleVisual.subjectNode, tripleVisual.objectNode].forEach(node => {
            node.scale.setScalar(0);
            const scaleAnimation = new THREE.VectorKeyframeTrack(
                '.scale',
                [0, 0.5],
                [0, 0, 0, 1, 1, 1]
            );
            
            const clip = new THREE.AnimationClip('scaleIn', 0.5, [scaleAnimation]);
            const action = this.animationMixer.clipAction(clip, node);
            action.setLoop(THREE.LoopOnce, 1);
            action.clampWhenFinished = true;
            action.play();
        });

        // Fade-in animation for edge
        const edgeMaterial = tripleVisual.predicateEdge.material as THREE.LineBasicMaterial;
        edgeMaterial.opacity = 0;
        
        let opacity = 0;
        const fadeIn = () => {
            opacity += 0.05;
            edgeMaterial.opacity = Math.min(0.7, opacity);
            
            if (opacity < 0.7) {
                requestAnimationFrame(fadeIn);
            }
        };
        fadeIn();

        // Particle burst effect
        if (this.config.renderEffects) {
            this.createInsertionParticleBurst(tripleVisual.subjectNode.position);
        }
    }

    private createInsertionParticleBurst(position: THREE.Vector3): void {
        const particleCount = 50;
        const positions = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = position.x;
            positions[i * 3 + 1] = position.y;
            positions[i * 3 + 2] = position.z;

            velocities[i * 3] = (Math.random() - 0.5) * 2;
            velocities[i * 3 + 1] = (Math.random() - 0.5) * 2;
            velocities[i * 3 + 2] = (Math.random() - 0.5) * 2;

            colors[i * 3] = Math.random() * 0.5 + 0.5;
            colors[i * 3 + 1] = Math.random() * 0.3 + 0.2;
            colors[i * 3 + 2] = Math.random() * 0.8 + 0.2;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true
        });

        const particles = new THREE.Points(geometry, material);
        this.scene.add(particles);

        // Animate burst
        let time = 0;
        const animate = () => {
            time += 0.016;
            const posAttr = particles.geometry.attributes.position as THREE.BufferAttribute;
            
            for (let i = 0; i < particleCount; i++) {
                posAttr.array[i * 3] += velocities[i * 3] * 0.1;
                posAttr.array[i * 3 + 1] += velocities[i * 3 + 1] * 0.1;
                posAttr.array[i * 3 + 2] += velocities[i * 3 + 2] * 0.1;
            }
            posAttr.needsUpdate = true;
            
            material.opacity = 1 - time / 2;
            
            if (time < 2) {
                requestAnimationFrame(animate);
            } else {
                this.scene.remove(particles);
                geometry.dispose();
                material.dispose();
            }
        };
        animate();
    }

    private updateTripleVisual(operation: TripleOperation): void {
        const existingVisual = this.triples.get(operation.triple.id);
        if (!existingVisual) return;

        // Update visual properties based on new triple data
        const triple = operation.triple;
        
        // Update colors based on confidence
        const confidenceAlpha = triple.confidence;
        [existingVisual.subjectNode, existingVisual.objectNode].forEach(node => {
            const material = node.material as THREE.MeshLambertMaterial;
            material.opacity = 0.8 * confidenceAlpha;
        });

        const edgeMaterial = existingVisual.predicateEdge.material as THREE.LineBasicMaterial;
        edgeMaterial.opacity = 0.7 * confidenceAlpha;

        // Pulse effect to indicate update
        if (this.config.highlightOperations) {
            this.animateTripleUpdate(existingVisual);
        }

        // Update spatial index
        if (this.config.spatialClustering) {
            this.updateSpatialIndex(triple);
        }
    }

    private animateTripleUpdate(tripleVisual: TripleVisual): void {
        // Pulse animation for nodes
        [tripleVisual.subjectNode, tripleVisual.objectNode].forEach(node => {
            const originalScale = node.scale.x;
            let scale = originalScale;
            let direction = 1;
            
            const pulse = () => {
                scale += direction * 0.05;
                if (scale > originalScale * 1.5) direction = -1;
                if (scale < originalScale) {
                    scale = originalScale;
                    node.scale.setScalar(scale);
                    return;
                }
                
                node.scale.setScalar(scale);
                requestAnimationFrame(pulse);
            };
            pulse();
        });

        // Color flash for edge
        const edgeMaterial = tripleVisual.predicateEdge.material as THREE.LineBasicMaterial;
        const originalColor = edgeMaterial.color.clone();
        edgeMaterial.color.setHex(0xFFFFFF);
        
        setTimeout(() => {
            edgeMaterial.color.copy(originalColor);
        }, 200);
    }

    private deleteTripleVisual(operation: TripleOperation): void {
        const tripleVisual = this.triples.get(operation.triple.id);
        if (!tripleVisual) return;

        // Animate deletion
        if (this.config.highlightOperations) {
            this.animateTripleDeletion(tripleVisual);
        }

        // Remove from scene after animation
        setTimeout(() => {
            this.scene.remove(tripleVisual.subjectNode);
            this.scene.remove(tripleVisual.predicateEdge);
            this.scene.remove(tripleVisual.objectNode);
            this.scene.remove(tripleVisual.predicateLabel);
            if (tripleVisual.connectionLine) {
                this.scene.remove(tripleVisual.connectionLine);
            }
            
            this.triples.delete(operation.triple.id);
        }, this.config.animationDuration);
    }

    private animateTripleDeletion(tripleVisual: TripleVisual): void {
        // Fade out and shrink animation
        [tripleVisual.subjectNode, tripleVisual.objectNode].forEach(node => {
            const material = node.material as THREE.MeshLambertMaterial;
            let opacity = material.opacity;
            let scale = node.scale.x;
            
            const fade = () => {
                opacity -= 0.02;
                scale -= 0.02;
                
                material.opacity = Math.max(0, opacity);
                node.scale.setScalar(Math.max(0, scale));
                
                if (opacity > 0 && scale > 0) {
                    requestAnimationFrame(fade);
                }
            };
            fade();
        });

        // Fade out edge
        const edgeMaterial = tripleVisual.predicateEdge.material as THREE.LineBasicMaterial;
        let edgeOpacity = edgeMaterial.opacity;
        
        const fadeEdge = () => {
            edgeOpacity -= 0.02;
            edgeMaterial.opacity = Math.max(0, edgeOpacity);
            
            if (edgeOpacity > 0) {
                requestAnimationFrame(fadeEdge);
            }
        };
        fadeEdge();

        // Particle dissolution effect
        if (this.config.renderEffects) {
            this.createDeletionParticleEffect(tripleVisual.subjectNode.position);
        }
    }

    private createDeletionParticleEffect(position: THREE.Vector3): void {
        const particleCount = 30;
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            const radius = Math.random() * 2;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;

            positions[i * 3] = position.x + radius * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = position.y + radius * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = position.z + radius * Math.cos(phi);

            colors[i * 3] = 0.8;
            colors[i * 3 + 1] = 0.2;
            colors[i * 3 + 2] = 0.2;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.03,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true
        });

        const particles = new THREE.Points(geometry, material);
        this.scene.add(particles);

        // Animate implosion
        let time = 0;
        const animate = () => {
            time += 0.016;
            const posAttr = particles.geometry.attributes.position as THREE.BufferAttribute;
            
            for (let i = 0; i < particleCount; i++) {
                const dx = position.x - posAttr.array[i * 3];
                const dy = position.y - posAttr.array[i * 3 + 1];
                const dz = position.z - posAttr.array[i * 3 + 2];
                
                posAttr.array[i * 3] += dx * 0.1;
                posAttr.array[i * 3 + 1] += dy * 0.1;
                posAttr.array[i * 3 + 2] += dz * 0.1;
            }
            posAttr.needsUpdate = true;
            
            if (time < 1) {
                requestAnimationFrame(animate);
            } else {
                this.scene.remove(particles);
                geometry.dispose();
                material.dispose();
            }
        };
        animate();
    }

    private queryTriples(operation: TripleOperation): void {
        // Highlight matching triples for query visualization
        const matchingTriples = this.findMatchingTriples(operation.triple);
        
        matchingTriples.forEach(tripleVisual => {
            this.highlightTripleForQuery(tripleVisual);
        });

        // Clear highlights after delay
        setTimeout(() => {
            matchingTriples.forEach(tripleVisual => {
                this.clearTripleHighlight(tripleVisual);
            });
        }, 2000);
    }

    private findMatchingTriples(queryTriple: Triple): TripleVisual[] {
        const matches: TripleVisual[] = [];
        
        this.triples.forEach(tripleVisual => {
            const triple = tripleVisual.triple;
            
            const subjectMatch = !queryTriple.subject || queryTriple.subject === '*' || triple.subject === queryTriple.subject;
            const predicateMatch = !queryTriple.predicate || queryTriple.predicate === '*' || triple.predicate === queryTriple.predicate;
            const objectMatch = !queryTriple.object || queryTriple.object === '*' || triple.object === queryTriple.object;
            
            if (subjectMatch && predicateMatch && objectMatch) {
                matches.push(tripleVisual);
            }
        });
        
        return matches;
    }

    private highlightTripleForQuery(tripleVisual: TripleVisual): void {
        [tripleVisual.subjectNode, tripleVisual.objectNode].forEach(node => {
            const material = node.material as THREE.MeshLambertMaterial;
            material.emissive.setHex(0x444400);
        });

        const edgeMaterial = tripleVisual.predicateEdge.material as THREE.LineBasicMaterial;
        edgeMaterial.color.setHex(0xFFFF00);
    }

    private clearTripleHighlight(tripleVisual: TripleVisual): void {
        [tripleVisual.subjectNode, tripleVisual.objectNode].forEach(node => {
            const material = node.material as THREE.MeshLambertMaterial;
            material.emissive.setHex(0x000000);
        });

        const edgeMaterial = tripleVisual.predicateEdge.material as THREE.LineBasicMaterial;
        edgeMaterial.color.copy(new THREE.Color(this.getPredicateColor(tripleVisual.triple.predicate)));
    }

    private visualizeTransaction(transaction: TripleTransaction): void {
        // Calculate bounding box for transaction operations
        const positions = transaction.operations
            .map(op => this.triples.get(op.triple.id))
            .filter(Boolean)
            .flatMap(tv => [tv!.subjectNode.position, tv!.objectNode.position]);

        if (positions.length === 0) return;

        const box = new THREE.Box3().setFromPoints(positions);
        const boxHelper = new THREE.Box3Helper(box, 0x00FF00);
        
        // Create progress ring
        const ringGeometry = new THREE.RingGeometry(2, 2.2, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: transaction.atomic ? 0xFF0000 : 0x00FF00,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide
        });
        const progressRing = new THREE.Mesh(ringGeometry, ringMaterial);
        progressRing.position.copy(box.getCenter(new THREE.Vector3()));

        // Create status indicator
        const statusGeometry = new THREE.SphereGeometry(0.2, 8, 8);
        const statusMaterial = new THREE.MeshBasicMaterial({
            color: this.getTransactionStatusColor(transaction.status)
        });
        const statusIndicator = new THREE.Mesh(statusGeometry, statusMaterial);
        statusIndicator.position.copy(box.getCenter(new THREE.Vector3()));
        statusIndicator.position.y += 3;

        const transactionVisual: TransactionVisual = {
            transaction,
            boundingBox: boxHelper,
            operationMarkers: [],
            progressRing,
            statusIndicator
        };

        this.scene.add(boxHelper);
        this.scene.add(progressRing);
        this.scene.add(statusIndicator);

        this.transactions.set(transaction.id, transactionVisual);

        // Animate transaction progress
        this.animateTransactionProgress(transactionVisual);
    }

    private getTransactionStatusColor(status: string): number {
        switch (status) {
            case 'pending': return 0xFFFF00;
            case 'committed': return 0x00FF00;
            case 'rolled_back': return 0xFF0000;
            default: return 0x888888;
        }
    }

    private animateTransactionProgress(transactionVisual: TransactionVisual): void {
        const { progressRing, transaction } = transactionVisual;
        const duration = 2000; // 2 seconds
        let progress = 0;

        const animate = () => {
            progress += 16 / duration; // Assuming 60fps
            
            // Update ring scale to show progress
            const scale = 1 + progress * 0.5;
            progressRing.scale.setScalar(scale);
            
            // Update opacity
            const material = progressRing.material as THREE.MeshBasicMaterial;
            material.opacity = 0.6 * (1 - progress);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                // Transaction complete
                transactionVisual.transaction.status = 'committed';
                const statusMaterial = transactionVisual.statusIndicator.material as THREE.MeshBasicMaterial;
                statusMaterial.color.setHex(this.getTransactionStatusColor('committed'));
            }
        };
        animate();
    }

    private updateSpatialIndex(triple: Triple): void {
        const position = this.getSpatialPosition(triple);
        this.spatialIndex.set(triple.id, position);
    }

    private addAtomicEffect(tripleVisual: TripleVisual, operationType: string): void {
        // Create atomic boundary effect
        const positions = [
            tripleVisual.subjectNode.position,
            tripleVisual.objectNode.position
        ];
        
        const center = new THREE.Vector3().addVectors(positions[0], positions[1]).multiplyScalar(0.5);
        const radius = positions[0].distanceTo(positions[1]) / 2 + 1;

        const atomicGeometry = new THREE.SphereGeometry(radius, 16, 16);
        const atomicMaterial = new THREE.MeshBasicMaterial({
            color: operationType === 'insert' ? 0x00FF00 : 0xFF0000,
            transparent: true,
            opacity: 0.1,
            wireframe: true
        });

        const atomicSphere = new THREE.Mesh(atomicGeometry, atomicMaterial);
        atomicSphere.position.copy(center);
        
        this.atomicEffects.add(atomicSphere);

        // Remove after short duration
        setTimeout(() => {
            this.atomicEffects.remove(atomicSphere);
            atomicGeometry.dispose();
            atomicMaterial.dispose();
        }, 1000);
    }

    private startAnimationLoop(): void {
        const animate = () => {
            const deltaTime = this.clock.getDelta();
            this.animationMixer.update(deltaTime);

            // Update particle system
            if (this.particleSystem) {
                this.particleSystem.rotation.x += 0.0005;
                this.particleSystem.rotation.y += 0.001;
                
                const positions = this.particleSystem.geometry.attributes.position as THREE.BufferAttribute;
                const colors = this.particleSystem.geometry.attributes.color as THREE.BufferAttribute;
                
                for (let i = 0; i < positions.count; i++) {
                    colors.setXYZ(
                        i,
                        colors.getX(i) + (Math.random() - 0.5) * 0.01,
                        colors.getY(i) + (Math.random() - 0.5) * 0.01,
                        colors.getZ(i) + (Math.random() - 0.5) * 0.01
                    );
                }
                colors.needsUpdate = true;
            }

            // Update atomic effects rotation
            this.atomicEffects.children.forEach((effect, index) => {
                effect.rotation.x += 0.01 * (index + 1);
                effect.rotation.y += 0.005 * (index + 1);
            });

            requestAnimationFrame(animate);
        };
        animate();
    }

    public getTripleCount(): number {
        return this.triples.size;
    }

    public getTransactionCount(): number {
        return this.transactions.size;
    }

    public exportTriples(): Triple[] {
        return Array.from(this.triples.values()).map(tv => tv.triple);
    }

    public setLayout(layout: VisualizationConfig['tripleLayout']): void {
        this.config.tripleLayout = layout;
        this.recomputeAllLayouts();
    }

    private recomputeAllLayouts(): void {
        this.triples.forEach((tripleVisual, tripleId) => {
            const newLayout = this.calculateTripleLayout(tripleVisual.triple);
            
            // Animate to new positions
            this.animateToNewLayout(tripleVisual, newLayout);
        });
    }

    private animateToNewLayout(tripleVisual: TripleVisual, newLayout: any): void {
        // Animate subject node
        const subjectAnimation = new THREE.VectorKeyframeTrack(
            '.position',
            [0, 1],
            [
                tripleVisual.subjectNode.position.x, tripleVisual.subjectNode.position.y, tripleVisual.subjectNode.position.z,
                newLayout.subjectPosition.x, newLayout.subjectPosition.y, newLayout.subjectPosition.z
            ]
        );

        const subjectClip = new THREE.AnimationClip('moveSubject', 1, [subjectAnimation]);
        const subjectAction = this.animationMixer.clipAction(subjectClip, tripleVisual.subjectNode);
        subjectAction.setLoop(THREE.LoopOnce, 1);
        subjectAction.clampWhenFinished = true;
        subjectAction.play();

        // Similar animations for object node and predicate label
        const objectAnimation = new THREE.VectorKeyframeTrack(
            '.position',
            [0, 1],
            [
                tripleVisual.objectNode.position.x, tripleVisual.objectNode.position.y, tripleVisual.objectNode.position.z,
                newLayout.objectPosition.x, newLayout.objectPosition.y, newLayout.objectPosition.z
            ]
        );

        const objectClip = new THREE.AnimationClip('moveObject', 1, [objectAnimation]);
        const objectAction = this.animationMixer.clipAction(objectClip, tripleVisual.objectNode);
        objectAction.setLoop(THREE.LoopOnce, 1);
        objectAction.clampWhenFinished = true;
        objectAction.play();
    }

    public clear(): void {
        this.triples.forEach((tripleVisual, tripleId) => {
            this.scene.remove(tripleVisual.subjectNode);
            this.scene.remove(tripleVisual.predicateEdge);
            this.scene.remove(tripleVisual.objectNode);
            this.scene.remove(tripleVisual.predicateLabel);
            if (tripleVisual.connectionLine) {
                this.scene.remove(tripleVisual.connectionLine);
            }
        });

        this.transactions.forEach((transactionVisual, transactionId) => {
            this.scene.remove(transactionVisual.boundingBox);
            this.scene.remove(transactionVisual.progressRing);
            this.scene.remove(transactionVisual.statusIndicator);
        });

        this.triples.clear();
        this.transactions.clear();
        this.operationQueue.length = 0;
        this.spatialIndex.clear();
    }

    public dispose(): void {
        this.clear();
        
        this.animationMixer.stopAllAction();
        
        if (this.particleSystem) {
            this.scene.remove(this.particleSystem);
            this.particleSystem.geometry.dispose();
            (this.particleSystem.material as THREE.Material).dispose();
        }

        this.scene.remove(this.atomicEffects);
        this.atomicEffects.clear();
    }
}