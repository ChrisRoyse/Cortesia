import * as THREE from 'three';
import { KnowledgeGraphAnimator, GraphNode, GraphEdge } from './KnowledgeGraphAnimator';

export interface EntityLifecycleEvent {
    id: string;
    timestamp: number;
    entityId: string;
    type: 'create' | 'update' | 'delete' | 'merge' | 'split';
    data: any;
    relatedEntities?: string[];
    metadata?: any;
}

export interface RelationshipEvent {
    id: string;
    timestamp: number;
    sourceEntity: string;
    targetEntity: string;
    relationType: string;
    type: 'form' | 'strengthen' | 'weaken' | 'dissolve';
    strength: number;
    confidence: number;
    metadata?: any;
}

export interface FlowVisualizationConfig {
    timeScale: number;
    flowSpeed: number;
    particleCount: number;
    trailLength: number;
    entityFadeTime: number;
    relationshipDecayRate: number;
    showHistory: boolean;
    colorByType: boolean;
    animateCreation: boolean;
}

export interface EntityState {
    id: string;
    position: THREE.Vector3;
    velocity: THREE.Vector3;
    size: number;
    strength: number;
    age: number;
    type: string;
    relationships: Set<string>;
    history: EntityLifecycleEvent[];
    particles?: THREE.Points;
    trail?: THREE.Line;
}

export interface RelationshipState {
    id: string;
    source: string;
    target: string;
    strength: number;
    age: number;
    decayRate: number;
    type: string;
    history: RelationshipEvent[];
    flowParticles?: THREE.Points;
    strengthIndicator?: THREE.Mesh;
}

export class EntityRelationshipFlow {
    private graphAnimator: KnowledgeGraphAnimator;
    private config: FlowVisualizationConfig;
    private entities: Map<string, EntityState> = new Map();
    private relationships: Map<string, RelationshipState> = new Map();
    private eventQueue: (EntityLifecycleEvent | RelationshipEvent)[] = [];
    private currentTime: number = 0;
    private animationId: number | null = null;
    private scene: THREE.Scene;
    private particlePool: THREE.Points[] = [];
    private trailPool: THREE.Line[] = [];

    constructor(
        graphAnimator: KnowledgeGraphAnimator,
        config?: Partial<FlowVisualizationConfig>
    ) {
        this.graphAnimator = graphAnimator;
        this.config = {
            timeScale: 1.0,
            flowSpeed: 2.0,
            particleCount: 100,
            trailLength: 20,
            entityFadeTime: 5000,
            relationshipDecayRate: 0.98,
            showHistory: true,
            colorByType: true,
            animateCreation: true,
            ...config
        };

        this.scene = (this.graphAnimator as any).scene;
        this.initializeParticlePools();
        this.startAnimation();
    }

    private initializeParticlePools(): void {
        // Pre-create particle systems for better performance
        for (let i = 0; i < 10; i++) {
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(this.config.particleCount * 3);
            const colors = new Float32Array(this.config.particleCount * 3);
            const sizes = new Float32Array(this.config.particleCount);
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

            const material = new THREE.PointsMaterial({
                size: 0.1,
                vertexColors: true,
                blending: THREE.AdditiveBlending,
                transparent: true,
                alphaTest: 0.01
            });

            const particles = new THREE.Points(geometry, material);
            this.particlePool.push(particles);
        }
    }

    public addEntityEvent(event: EntityLifecycleEvent): void {
        this.eventQueue.push(event);
        this.eventQueue.sort((a, b) => a.timestamp - b.timestamp);
    }

    public addRelationshipEvent(event: RelationshipEvent): void {
        this.eventQueue.push(event);
        this.eventQueue.sort((a, b) => a.timestamp - b.timestamp);
    }

    private processEvents(): void {
        const currentTimeWindow = this.currentTime + 100; // Process events in 100ms window
        
        while (this.eventQueue.length > 0 && this.eventQueue[0].timestamp <= currentTimeWindow) {
            const event = this.eventQueue.shift()!;
            
            if ('entityId' in event) {
                this.processEntityEvent(event);
            } else {
                this.processRelationshipEvent(event);
            }
        }
    }

    private processEntityEvent(event: EntityLifecycleEvent): void {
        switch (event.type) {
            case 'create':
                this.createEntity(event);
                break;
            case 'update':
                this.updateEntity(event);
                break;
            case 'delete':
                this.deleteEntity(event);
                break;
            case 'merge':
                this.mergeEntities(event);
                break;
            case 'split':
                this.splitEntity(event);
                break;
        }
    }

    private createEntity(event: EntityLifecycleEvent): void {
        const node = this.graphAnimator.getNode(event.entityId);
        if (!node) return;

        const entityState: EntityState = {
            id: event.entityId,
            position: node.position.clone(),
            velocity: new THREE.Vector3(0, 0, 0),
            size: 1.0,
            strength: 1.0,
            age: 0,
            type: event.data.type || 'entity',
            relationships: new Set(),
            history: [event]
        };

        // Create birth animation
        if (this.config.animateCreation) {
            this.animateEntityCreation(entityState);
        }

        // Create entity particles
        entityState.particles = this.createEntityParticles(entityState);
        if (entityState.particles) {
            this.scene.add(entityState.particles);
        }

        // Create history trail
        if (this.config.showHistory) {
            entityState.trail = this.createEntityTrail(entityState);
            if (entityState.trail) {
                this.scene.add(entityState.trail);
            }
        }

        this.entities.set(event.entityId, entityState);
    }

    private updateEntity(event: EntityLifecycleEvent): void {
        const entityState = this.entities.get(event.entityId);
        if (!entityState) return;

        entityState.history.push(event);
        
        // Update entity properties based on event data
        if (event.data.strength !== undefined) {
            entityState.strength = event.data.strength;
        }
        
        if (event.data.size !== undefined) {
            entityState.size = event.data.size;
        }

        // Animate update with pulse effect
        this.animateEntityUpdate(entityState, event);
    }

    private deleteEntity(event: EntityLifecycleEvent): void {
        const entityState = this.entities.get(event.entityId);
        if (!entityState) return;

        // Animate dissolution
        this.animateEntityDeletion(entityState);

        // Clean up after fade animation
        setTimeout(() => {
            if (entityState.particles) this.scene.remove(entityState.particles);
            if (entityState.trail) this.scene.remove(entityState.trail);
            this.entities.delete(event.entityId);
        }, this.config.entityFadeTime);
    }

    private mergeEntities(event: EntityLifecycleEvent): void {
        if (!event.relatedEntities || event.relatedEntities.length < 2) return;

        const sourceEntities = event.relatedEntities.map(id => this.entities.get(id)).filter(Boolean);
        const targetEntity = this.entities.get(event.entityId);

        if (!targetEntity || sourceEntities.length === 0) return;

        // Animate merge process
        sourceEntities.forEach(sourceEntity => {
            if (sourceEntity) {
                this.animateEntityMerge(sourceEntity, targetEntity);
            }
        });

        // Update target entity properties
        targetEntity.strength += sourceEntities.reduce((sum, e) => sum + (e?.strength || 0), 0);
        targetEntity.size = Math.max(targetEntity.size, Math.max(...sourceEntities.map(e => e?.size || 1)));
    }

    private splitEntity(event: EntityLifecycleEvent): void {
        const sourceEntity = this.entities.get(event.entityId);
        if (!sourceEntity || !event.relatedEntities) return;

        // Create split animation
        event.relatedEntities.forEach((newEntityId, index) => {
            const angle = (index / event.relatedEntities!.length) * Math.PI * 2;
            const offset = new THREE.Vector3(
                Math.cos(angle) * 2,
                Math.sin(angle) * 2,
                0
            );

            // Create new entity at offset position
            const splitEvent: EntityLifecycleEvent = {
                id: `split_${newEntityId}`,
                timestamp: event.timestamp,
                entityId: newEntityId,
                type: 'create',
                data: {
                    type: sourceEntity.type,
                    strength: sourceEntity.strength / event.relatedEntities!.length,
                    size: sourceEntity.size * 0.8
                }
            };

            this.createEntity(splitEvent);

            // Animate movement from source to new position
            const newEntity = this.entities.get(newEntityId);
            if (newEntity) {
                newEntity.position.copy(sourceEntity.position);
                this.animateEntitySplit(newEntity, offset);
            }
        });

        // Reduce source entity
        sourceEntity.strength *= 0.5;
        sourceEntity.size *= 0.8;
    }

    private processRelationshipEvent(event: RelationshipEvent): void {
        const relationshipId = `${event.sourceEntity}-${event.targetEntity}-${event.relationType}`;
        
        switch (event.type) {
            case 'form':
                this.formRelationship(relationshipId, event);
                break;
            case 'strengthen':
                this.strengthenRelationship(relationshipId, event);
                break;
            case 'weaken':
                this.weakenRelationship(relationshipId, event);
                break;
            case 'dissolve':
                this.dissolveRelationship(relationshipId, event);
                break;
        }
    }

    private formRelationship(relationshipId: string, event: RelationshipEvent): void {
        const sourceEntity = this.entities.get(event.sourceEntity);
        const targetEntity = this.entities.get(event.targetEntity);
        
        if (!sourceEntity || !targetEntity) return;

        const relationshipState: RelationshipState = {
            id: relationshipId,
            source: event.sourceEntity,
            target: event.targetEntity,
            strength: event.strength,
            age: 0,
            decayRate: this.config.relationshipDecayRate,
            type: event.relationType,
            history: [event]
        };

        // Create flow particles between entities
        relationshipState.flowParticles = this.createRelationshipFlow(sourceEntity, targetEntity, relationshipState);
        if (relationshipState.flowParticles) {
            this.scene.add(relationshipState.flowParticles);
        }

        // Create strength indicator
        relationshipState.strengthIndicator = this.createStrengthIndicator(relationshipState);
        if (relationshipState.strengthIndicator) {
            this.scene.add(relationshipState.strengthIndicator);
        }

        // Update entity relationship sets
        sourceEntity.relationships.add(relationshipId);
        targetEntity.relationships.add(relationshipId);

        this.relationships.set(relationshipId, relationshipState);

        // Animate relationship formation
        this.animateRelationshipFormation(relationshipState);
    }

    private strengthenRelationship(relationshipId: string, event: RelationshipEvent): void {
        const relationship = this.relationships.get(relationshipId);
        if (!relationship) return;

        relationship.strength = Math.min(1.0, relationship.strength + 0.1);
        relationship.history.push(event);

        // Animate strengthening with brighter flow
        this.animateRelationshipStrengthening(relationship);
    }

    private weakenRelationship(relationshipId: string, event: RelationshipEvent): void {
        const relationship = this.relationships.get(relationshipId);
        if (!relationship) return;

        relationship.strength = Math.max(0.0, relationship.strength - 0.1);
        relationship.history.push(event);

        // Animate weakening with dimmer flow
        this.animateRelationshipWeakening(relationship);
    }

    private dissolveRelationship(relationshipId: string, event: RelationshipEvent): void {
        const relationship = this.relationships.get(relationshipId);
        if (!relationship) return;

        // Animate dissolution
        this.animateRelationshipDissolution(relationship);

        // Clean up
        setTimeout(() => {
            if (relationship.flowParticles) this.scene.remove(relationship.flowParticles);
            if (relationship.strengthIndicator) this.scene.remove(relationship.strengthIndicator);
            
            // Remove from entity relationship sets
            const sourceEntity = this.entities.get(relationship.source);
            const targetEntity = this.entities.get(relationship.target);
            if (sourceEntity) sourceEntity.relationships.delete(relationshipId);
            if (targetEntity) targetEntity.relationships.delete(relationshipId);
            
            this.relationships.delete(relationshipId);
        }, 1000);
    }

    private createEntityParticles(entityState: EntityState): THREE.Points {
        const particleCount = Math.floor(this.config.particleCount * entityState.strength);
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);

        const color = this.getEntityTypeColor(entityState.type);
        
        for (let i = 0; i < particleCount; i++) {
            // Random position around entity
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const radius = Math.random() * entityState.size;
            
            positions[i * 3] = entityState.position.x + radius * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = entityState.position.y + radius * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = entityState.position.z + radius * Math.cos(phi);
            
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
            
            sizes[i] = Math.random() * 0.1 + 0.05;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        const material = new THREE.PointsMaterial({
            size: 0.1,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true,
            alphaTest: 0.01
        });

        return new THREE.Points(geometry, material);
    }

    private createEntityTrail(entityState: EntityState): THREE.Line {
        const points = [];
        for (let i = 0; i < this.config.trailLength; i++) {
            points.push(entityState.position.clone());
        }

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: this.getEntityTypeColor(entityState.type),
            transparent: true,
            opacity: 0.3
        });

        return new THREE.Line(geometry, material);
    }

    private createRelationshipFlow(
        sourceEntity: EntityState,
        targetEntity: EntityState,
        relationship: RelationshipState
    ): THREE.Points {
        const particleCount = Math.floor(50 * relationship.strength);
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);

        const color = this.getRelationshipTypeColor(relationship.type);
        
        for (let i = 0; i < particleCount; i++) {
            const t = i / particleCount;
            const pos = sourceEntity.position.clone().lerp(targetEntity.position, t);
            
            positions[i * 3] = pos.x;
            positions[i * 3 + 1] = pos.y;
            positions[i * 3 + 2] = pos.z;
            
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
            
            sizes[i] = Math.random() * 0.08 + 0.02;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true,
            alphaTest: 0.01
        });

        return new THREE.Points(geometry, material);
    }

    private createStrengthIndicator(relationship: RelationshipState): THREE.Mesh {
        const geometry = new THREE.CylinderGeometry(0.02, 0.02, 1, 8);
        const material = new THREE.MeshBasicMaterial({
            color: this.getRelationshipTypeColor(relationship.type),
            transparent: true,
            opacity: relationship.strength
        });

        const indicator = new THREE.Mesh(geometry, material);
        
        // Position at midpoint of relationship
        const sourceEntity = this.entities.get(relationship.source);
        const targetEntity = this.entities.get(relationship.target);
        if (sourceEntity && targetEntity) {
            indicator.position.copy(sourceEntity.position).lerp(targetEntity.position, 0.5);
        }

        return indicator;
    }

    private getEntityTypeColor(type: string): THREE.Color {
        if (!this.config.colorByType) return new THREE.Color(0x4a90e2);

        const colors: { [key: string]: number } = {
            person: 0x4CAF50,
            place: 0x2196F3,
            thing: 0xFF9800,
            concept: 0x9C27B0,
            event: 0xF44336,
            default: 0x4a90e2
        };

        return new THREE.Color(colors[type] || colors.default);
    }

    private getRelationshipTypeColor(type: string): THREE.Color {
        const colors: { [key: string]: number } = {
            knows: 0x4CAF50,
            related_to: 0x2196F3,
            part_of: 0xFF9800,
            similar_to: 0x9C27B0,
            caused_by: 0xF44336,
            default: 0x666666
        };

        return new THREE.Color(colors[type] || colors.default);
    }

    private animateEntityCreation(entityState: EntityState): void {
        // Burst effect on creation
        if (entityState.particles) {
            const positions = entityState.particles.geometry.attributes.position as THREE.BufferAttribute;
            const originalPositions = positions.array.slice();
            
            // Animate particles expanding and contracting
            let progress = 0;
            const animate = () => {
                progress += 0.05;
                const scale = 1 + Math.sin(progress * Math.PI) * 2;
                
                for (let i = 0; i < positions.count; i++) {
                    positions.setXYZ(
                        i,
                        originalPositions[i * 3] * scale,
                        originalPositions[i * 3 + 1] * scale,
                        originalPositions[i * 3 + 2] * scale
                    );
                }
                positions.needsUpdate = true;
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                }
            };
            animate();
        }
    }

    private animateEntityUpdate(entityState: EntityState, event: EntityLifecycleEvent): void {
        // Pulse effect on update
        const node = this.graphAnimator.getNode(entityState.id);
        if (node) {
            this.graphAnimator.updateNodeActivation(entityState.id, 1.0);
            setTimeout(() => {
                this.graphAnimator.updateNodeActivation(entityState.id, 0.0);
            }, 500);
        }
    }

    private animateEntityDeletion(entityState: EntityState): void {
        // Fade out animation
        if (entityState.particles) {
            const material = entityState.particles.material as THREE.PointsMaterial;
            let opacity = material.opacity || 1.0;
            
            const fade = () => {
                opacity -= 0.02;
                material.opacity = Math.max(0, opacity);
                
                if (opacity > 0) {
                    requestAnimationFrame(fade);
                }
            };
            fade();
        }
    }

    private animateEntityMerge(sourceEntity: EntityState, targetEntity: EntityState): void {
        // Move source particles to target
        if (sourceEntity.particles) {
            const positions = sourceEntity.particles.geometry.attributes.position as THREE.BufferAttribute;
            
            let progress = 0;
            const animate = () => {
                progress += 0.02;
                
                for (let i = 0; i < positions.count; i++) {
                    const currentPos = new THREE.Vector3(
                        positions.getX(i),
                        positions.getY(i),
                        positions.getZ(i)
                    );
                    
                    const targetPos = targetEntity.position.clone();
                    currentPos.lerp(targetPos, progress);
                    
                    positions.setXYZ(i, currentPos.x, currentPos.y, currentPos.z);
                }
                positions.needsUpdate = true;
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                }
            };
            animate();
        }
    }

    private animateEntitySplit(newEntity: EntityState, offset: THREE.Vector3): void {
        const targetPosition = newEntity.position.clone().add(offset);
        
        let progress = 0;
        const animate = () => {
            progress += 0.02;
            newEntity.position.lerp(targetPosition, progress);
            
            if (newEntity.particles) {
                const positions = newEntity.particles.geometry.attributes.position as THREE.BufferAttribute;
                for (let i = 0; i < positions.count; i++) {
                    positions.setXYZ(
                        i,
                        newEntity.position.x + (Math.random() - 0.5) * newEntity.size,
                        newEntity.position.y + (Math.random() - 0.5) * newEntity.size,
                        newEntity.position.z + (Math.random() - 0.5) * newEntity.size
                    );
                }
                positions.needsUpdate = true;
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        animate();
    }

    private animateRelationshipFormation(relationship: RelationshipState): void {
        if (relationship.flowParticles) {
            const material = relationship.flowParticles.material as THREE.PointsMaterial;
            material.opacity = 0;
            
            let opacity = 0;
            const fade = () => {
                opacity += 0.05;
                material.opacity = Math.min(1, opacity);
                
                if (opacity < 1) {
                    requestAnimationFrame(fade);
                }
            };
            fade();
        }
    }

    private animateRelationshipStrengthening(relationship: RelationshipState): void {
        if (relationship.strengthIndicator) {
            const material = relationship.strengthIndicator.material as THREE.MeshBasicMaterial;
            material.opacity = relationship.strength;
            
            // Brief flash effect
            material.emissive.setHex(0x222222);
            setTimeout(() => {
                material.emissive.setHex(0x000000);
            }, 200);
        }
    }

    private animateRelationshipWeakening(relationship: RelationshipState): void {
        if (relationship.strengthIndicator) {
            const material = relationship.strengthIndicator.material as THREE.MeshBasicMaterial;
            material.opacity = relationship.strength;
        }
    }

    private animateRelationshipDissolution(relationship: RelationshipState): void {
        if (relationship.flowParticles) {
            const material = relationship.flowParticles.material as THREE.PointsMaterial;
            let opacity = material.opacity || 1.0;
            
            const fade = () => {
                opacity -= 0.05;
                material.opacity = Math.max(0, opacity);
                
                if (opacity > 0) {
                    requestAnimationFrame(fade);
                }
            };
            fade();
        }
    }

    private updateEntityStates(): void {
        this.entities.forEach((entityState, entityId) => {
            entityState.age += 1;
            
            // Update particle positions for flowing effect
            if (entityState.particles) {
                const positions = entityState.particles.geometry.attributes.position as THREE.BufferAttribute;
                const colors = entityState.particles.geometry.attributes.color as THREE.BufferAttribute;
                
                for (let i = 0; i < positions.count; i++) {
                    // Add slight random movement
                    const x = positions.getX(i) + (Math.random() - 0.5) * 0.01;
                    const y = positions.getY(i) + (Math.random() - 0.5) * 0.01;
                    const z = positions.getZ(i) + (Math.random() - 0.5) * 0.01;
                    
                    positions.setXYZ(i, x, y, z);
                    
                    // Pulse color based on entity strength
                    const intensity = 0.5 + 0.5 * Math.sin(this.currentTime * 0.001 + i * 0.1) * entityState.strength;
                    colors.setXYZ(i, colors.getX(i) * intensity, colors.getY(i) * intensity, colors.getZ(i) * intensity);
                }
                
                positions.needsUpdate = true;
                colors.needsUpdate = true;
            }
            
            // Update trail
            if (entityState.trail) {
                this.updateEntityTrail(entityState);
            }
        });
    }

    private updateRelationshipStates(): void {
        this.relationships.forEach((relationship, relationshipId) => {
            relationship.age += 1;
            relationship.strength *= relationship.decayRate;
            
            // Remove very weak relationships
            if (relationship.strength < 0.01) {
                this.dissolveRelationship(relationshipId, {
                    id: `auto_dissolve_${relationshipId}`,
                    timestamp: this.currentTime,
                    sourceEntity: relationship.source,
                    targetEntity: relationship.target,
                    relationType: relationship.type,
                    type: 'dissolve',
                    strength: 0,
                    confidence: 0
                });
                return;
            }
            
            // Update flow particles
            if (relationship.flowParticles) {
                this.updateRelationshipFlow(relationship);
            }
            
            // Update strength indicator
            if (relationship.strengthIndicator) {
                const material = relationship.strengthIndicator.material as THREE.MeshBasicMaterial;
                material.opacity = relationship.strength;
            }
        });
    }

    private updateEntityTrail(entityState: EntityState): void {
        if (!entityState.trail) return;
        
        const geometry = entityState.trail.geometry as THREE.BufferGeometry;
        const positions = geometry.attributes.position as THREE.BufferAttribute;
        
        // Shift trail points
        for (let i = positions.count - 1; i > 0; i--) {
            positions.setXYZ(
                i,
                positions.getX(i - 1),
                positions.getY(i - 1),
                positions.getZ(i - 1)
            );
        }
        
        // Add current position at front
        positions.setXYZ(0, entityState.position.x, entityState.position.y, entityState.position.z);
        positions.needsUpdate = true;
    }

    private updateRelationshipFlow(relationship: RelationshipState): void {
        const sourceEntity = this.entities.get(relationship.source);
        const targetEntity = this.entities.get(relationship.target);
        
        if (!sourceEntity || !targetEntity || !relationship.flowParticles) return;
        
        const positions = relationship.flowParticles.geometry.attributes.position as THREE.BufferAttribute;
        const particleCount = positions.count;
        
        for (let i = 0; i < particleCount; i++) {
            const t = (this.currentTime * this.config.flowSpeed * 0.001 + i / particleCount) % 1;
            const pos = sourceEntity.position.clone().lerp(targetEntity.position, t);
            
            positions.setXYZ(i, pos.x, pos.y, pos.z);
        }
        
        positions.needsUpdate = true;
    }

    private animate = (): void => {
        this.currentTime = performance.now();
        
        this.processEvents();
        this.updateEntityStates();
        this.updateRelationshipStates();
        
        this.animationId = requestAnimationFrame(this.animate);
    };

    private startAnimation(): void {
        if (!this.animationId) {
            this.animate();
        }
    }

    public stopAnimation(): void {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    public setTimeScale(scale: number): void {
        this.config.timeScale = Math.max(0.1, Math.min(10.0, scale));
    }

    public getEntityState(entityId: string): EntityState | undefined {
        return this.entities.get(entityId);
    }

    public getRelationshipState(relationshipId: string): RelationshipState | undefined {
        return this.relationships.get(relationshipId);
    }

    public exportFlowData(): any {
        return {
            entities: Array.from(this.entities.values()).map(entity => ({
                id: entity.id,
                type: entity.type,
                strength: entity.strength,
                age: entity.age,
                relationships: Array.from(entity.relationships),
                history: entity.history
            })),
            relationships: Array.from(this.relationships.values()).map(rel => ({
                id: rel.id,
                source: rel.source,
                target: rel.target,
                strength: rel.strength,
                age: rel.age,
                type: rel.type,
                history: rel.history
            }))
        };
    }

    public clear(): void {
        this.entities.forEach((entity, entityId) => {
            if (entity.particles) this.scene.remove(entity.particles);
            if (entity.trail) this.scene.remove(entity.trail);
        });
        
        this.relationships.forEach((relationship, relationshipId) => {
            if (relationship.flowParticles) this.scene.remove(relationship.flowParticles);
            if (relationship.strengthIndicator) this.scene.remove(relationship.strengthIndicator);
        });
        
        this.entities.clear();
        this.relationships.clear();
        this.eventQueue.length = 0;
    }

    public dispose(): void {
        this.stopAnimation();
        this.clear();
        
        this.particlePool.forEach(particles => {
            particles.geometry.dispose();
            (particles.material as THREE.Material).dispose();
        });
        this.particlePool.length = 0;
        
        this.trailPool.forEach(trail => {
            trail.geometry.dispose();
            (trail.material as THREE.Material).dispose();
        });
        this.trailPool.length = 0;
    }
}