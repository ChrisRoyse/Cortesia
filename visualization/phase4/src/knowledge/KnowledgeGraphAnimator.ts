import * as THREE from 'three';

export interface GraphNode {
    id: string;
    type: 'entity' | 'relation' | 'logic_gate' | 'hidden';
    position: THREE.Vector3;
    velocity: THREE.Vector3;
    mass: number;
    size: number;
    activation: number;
    color: THREE.Color;
    userData: any;
    mesh?: THREE.Mesh;
    label?: THREE.Sprite;
}

export interface GraphEdge {
    id: string;
    source: string;
    target: string;
    type: 'relationship' | 'inhibitory' | 'activation' | 'logic';
    strength: number;
    weight: number;
    color: THREE.Color;
    animated: boolean;
    line?: THREE.Line;
    particles?: THREE.Points;
}

export interface AnimationState {
    playing: boolean;
    speed: number;
    time: number;
    deltaTime: number;
}

export interface GraphConfig {
    nodeSize: { min: number; max: number };
    edgeWidth: { min: number; max: number };
    forceStrength: number;
    damping: number;
    centeringForce: number;
    repulsionForce: number;
    springLength: number;
    maxVelocity: number;
    layoutSteps: number;
}

export class KnowledgeGraphAnimator {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private renderer: THREE.WebGLRenderer;
    private nodes: Map<string, GraphNode> = new Map();
    private edges: Map<string, GraphEdge> = new Map();
    private animationState: AnimationState;
    private config: GraphConfig;
    private raycaster: THREE.Raycaster;
    private mouse: THREE.Vector2;
    private selectedNode: GraphNode | null = null;
    private hoveredNode: GraphNode | null = null;
    private animationId: number | null = null;
    private particleSystem: THREE.Points | null = null;
    private activationPaths: Map<string, THREE.Line[]> = new Map();

    constructor(container: HTMLElement, config?: Partial<GraphConfig>) {
        this.config = {
            nodeSize: { min: 0.5, max: 3.0 },
            edgeWidth: { min: 0.1, max: 0.8 },
            forceStrength: 100.0,
            damping: 0.85,
            centeringForce: 0.01,
            repulsionForce: 50.0,
            springLength: 5.0,
            maxVelocity: 2.0,
            layoutSteps: 10,
            ...config
        };

        this.animationState = {
            playing: true,
            speed: 1.0,
            time: 0,
            deltaTime: 0
        };

        this.initializeScene(container);
        this.setupEventHandlers();
        this.createParticleSystem();
        this.startAnimation();
    }

    private initializeScene(container: HTMLElement): void {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.Fog(0x0a0a0a, 10, 100);

        // Camera setup
        const aspect = container.clientWidth / container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 20);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true,
            powerPreference: 'high-performance'
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(this.renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.setScalar(2048);
        this.scene.add(directionalLight);

        // Setup raycaster for mouse interaction
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
    }

    private setupEventHandlers(): void {
        const canvas = this.renderer.domElement;

        canvas.addEventListener('mousemove', (event) => {
            const rect = canvas.getBoundingClientRect();
            this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            this.handleMouseMove();
        });

        canvas.addEventListener('click', (event) => {
            this.handleClick();
        });

        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }

    private createParticleSystem(): void {
        const particleCount = 1000;
        const particles = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount; i++) {
            particles[i * 3] = (Math.random() - 0.5) * 100;
            particles[i * 3 + 1] = (Math.random() - 0.5) * 100;
            particles[i * 3 + 2] = (Math.random() - 0.5) * 100;

            colors[i * 3] = Math.random() * 0.5 + 0.5;
            colors[i * 3 + 1] = Math.random() * 0.3 + 0.2;
            colors[i * 3 + 2] = Math.random() * 0.8 + 0.2;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(particles, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.02,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true,
            opacity: 0.6
        });

        this.particleSystem = new THREE.Points(geometry, material);
        this.scene.add(this.particleSystem);
    }

    public addNode(nodeData: Partial<GraphNode> & { id: string }): GraphNode {
        const node: GraphNode = {
            type: 'entity',
            position: new THREE.Vector3(
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20
            ),
            velocity: new THREE.Vector3(0, 0, 0),
            mass: 1.0,
            size: 1.0,
            activation: 0.0,
            color: new THREE.Color(0x4a90e2),
            userData: {},
            ...nodeData
        };

        // Create node mesh
        const geometry = this.getNodeGeometry(node.type);
        const material = new THREE.MeshLambertMaterial({ 
            color: node.color,
            transparent: true,
            opacity: 0.8
        });
        
        node.mesh = new THREE.Mesh(geometry, material);
        node.mesh.position.copy(node.position);
        node.mesh.scale.setScalar(node.size);
        node.mesh.castShadow = true;
        node.mesh.receiveShadow = true;
        node.mesh.userData = { nodeId: node.id };
        
        this.scene.add(node.mesh);

        // Create label
        node.label = this.createNodeLabel(node.id);
        if (node.label) {
            this.scene.add(node.label);
        }

        this.nodes.set(node.id, node);
        return node;
    }

    public addEdge(edgeData: Partial<GraphEdge> & { id: string; source: string; target: string }): GraphEdge {
        const sourceNode = this.nodes.get(edgeData.source);
        const targetNode = this.nodes.get(edgeData.target);
        
        if (!sourceNode || !targetNode) {
            throw new Error(`Source or target node not found for edge ${edgeData.id}`);
        }

        const edge: GraphEdge = {
            type: 'relationship',
            strength: 1.0,
            weight: 1.0,
            color: new THREE.Color(0x666666),
            animated: false,
            ...edgeData
        };

        // Create edge line
        const points = [sourceNode.position.clone(), targetNode.position.clone()];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ 
            color: edge.color,
            transparent: true,
            opacity: 0.6
        });
        
        edge.line = new THREE.Line(geometry, material);
        this.scene.add(edge.line);

        // Create particle flow for animated edges
        if (edge.animated) {
            this.createEdgeParticles(edge);
        }

        this.edges.set(edge.id, edge);
        return edge;
    }

    private getNodeGeometry(type: GraphNode['type']): THREE.BufferGeometry {
        switch (type) {
            case 'entity':
                return new THREE.SphereGeometry(1, 16, 16);
            case 'relation':
                return new THREE.BoxGeometry(1, 1, 1);
            case 'logic_gate':
                return new THREE.OctahedronGeometry(1, 0);
            case 'hidden':
                return new THREE.TetrahedronGeometry(1, 0);
            default:
                return new THREE.SphereGeometry(1, 16, 16);
        }
    }

    private createNodeLabel(text: string): THREE.Sprite {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d')!;
        canvas.width = 256;
        canvas.height = 64;
        
        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        context.fillStyle = 'white';
        context.font = '20px Arial';
        context.textAlign = 'center';
        context.fillText(text, canvas.width / 2, canvas.height / 2 + 7);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ 
            map: texture,
            transparent: true,
            opacity: 0.8
        });
        
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(2, 0.5, 1);
        return sprite;
    }

    private createEdgeParticles(edge: GraphEdge): void {
        const particleCount = 20;
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = 0;
            positions[i * 3 + 1] = 0;
            positions[i * 3 + 2] = 0;

            colors[i * 3] = edge.color.r;
            colors[i * 3 + 1] = edge.color.g;
            colors[i * 3 + 2] = edge.color.b;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.1,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true
        });

        edge.particles = new THREE.Points(geometry, material);
        this.scene.add(edge.particles);
    }

    private updateForces(): void {
        // Reset forces
        this.nodes.forEach(node => {
            node.velocity.multiplyScalar(this.config.damping);
        });

        // Apply repulsion forces between nodes
        const nodeArray = Array.from(this.nodes.values());
        for (let i = 0; i < nodeArray.length; i++) {
            for (let j = i + 1; j < nodeArray.length; j++) {
                const node1 = nodeArray[i];
                const node2 = nodeArray[j];
                
                const direction = node1.position.clone().sub(node2.position);
                const distance = direction.length();
                
                if (distance > 0) {
                    const force = this.config.repulsionForce / (distance * distance);
                    direction.normalize().multiplyScalar(force);
                    
                    node1.velocity.add(direction.clone().multiplyScalar(1 / node1.mass));
                    node2.velocity.add(direction.clone().multiplyScalar(-1 / node2.mass));
                }
            }
        }

        // Apply spring forces from edges
        this.edges.forEach(edge => {
            const sourceNode = this.nodes.get(edge.source);
            const targetNode = this.nodes.get(edge.target);
            
            if (sourceNode && targetNode) {
                const direction = targetNode.position.clone().sub(sourceNode.position);
                const distance = direction.length();
                const displacement = distance - this.config.springLength;
                
                if (distance > 0) {
                    const force = displacement * edge.strength * 0.1;
                    direction.normalize().multiplyScalar(force);
                    
                    sourceNode.velocity.add(direction.clone().multiplyScalar(1 / sourceNode.mass));
                    targetNode.velocity.add(direction.clone().multiplyScalar(-1 / targetNode.mass));
                }
            }
        });

        // Apply centering force
        this.nodes.forEach(node => {
            const centerForce = node.position.clone().multiplyScalar(-this.config.centeringForce);
            node.velocity.add(centerForce.multiplyScalar(1 / node.mass));
        });
    }

    private updatePositions(): void {
        this.nodes.forEach(node => {
            // Limit velocity
            if (node.velocity.length() > this.config.maxVelocity) {
                node.velocity.normalize().multiplyScalar(this.config.maxVelocity);
            }

            // Update position
            node.position.add(node.velocity.clone().multiplyScalar(this.animationState.deltaTime));

            // Update mesh position
            if (node.mesh) {
                node.mesh.position.copy(node.position);
                
                // Update size based on activation
                const scale = node.size * (1 + node.activation * 0.5);
                node.mesh.scale.setScalar(scale);

                // Update color based on activation
                const material = node.mesh.material as THREE.MeshLambertMaterial;
                const intensity = 1 + node.activation * 2;
                material.color.copy(node.color).multiplyScalar(intensity);
            }

            // Update label position
            if (node.label) {
                node.label.position.copy(node.position);
                node.label.position.y += node.size + 0.5;
            }
        });

        // Update edge lines
        this.edges.forEach(edge => {
            const sourceNode = this.nodes.get(edge.source);
            const targetNode = this.nodes.get(edge.target);
            
            if (sourceNode && targetNode && edge.line) {
                const points = [sourceNode.position.clone(), targetNode.position.clone()];
                edge.line.geometry.setFromPoints(points);
                edge.line.geometry.attributes.position.needsUpdate = true;
            }

            // Update edge particles
            if (edge.particles && sourceNode && targetNode) {
                this.updateEdgeParticles(edge, sourceNode, targetNode);
            }
        });
    }

    private updateEdgeParticles(edge: GraphEdge, sourceNode: GraphNode, targetNode: GraphNode): void {
        if (!edge.particles) return;

        const positions = edge.particles.geometry.attributes.position as THREE.BufferAttribute;
        const particleCount = positions.count;
        
        for (let i = 0; i < particleCount; i++) {
            const t = (this.animationState.time * edge.strength + i / particleCount) % 1;
            const pos = sourceNode.position.clone().lerp(targetNode.position, t);
            
            positions.setXYZ(i, pos.x, pos.y, pos.z);
        }
        
        positions.needsUpdate = true;
    }

    private handleMouseMove(): void {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(
            Array.from(this.nodes.values()).map(node => node.mesh).filter(Boolean) as THREE.Mesh[]
        );

        // Reset previous hover
        if (this.hoveredNode && this.hoveredNode.mesh) {
            const material = this.hoveredNode.mesh.material as THREE.MeshLambertMaterial;
            material.emissive.setHex(0x000000);
        }

        if (intersects.length > 0) {
            const nodeId = intersects[0].object.userData.nodeId;
            this.hoveredNode = this.nodes.get(nodeId) || null;
            
            if (this.hoveredNode && this.hoveredNode.mesh) {
                const material = this.hoveredNode.mesh.material as THREE.MeshLambertMaterial;
                material.emissive.setHex(0x222222);
            }
        } else {
            this.hoveredNode = null;
        }
    }

    private handleClick(): void {
        if (this.hoveredNode) {
            this.selectedNode = this.hoveredNode;
            this.highlightNodeConnections(this.selectedNode.id);
        } else {
            this.selectedNode = null;
            this.clearHighlights();
        }
    }

    private highlightNodeConnections(nodeId: string): void {
        this.clearHighlights();
        
        this.edges.forEach(edge => {
            if (edge.source === nodeId || edge.target === nodeId) {
                if (edge.line) {
                    const material = edge.line.material as THREE.LineBasicMaterial;
                    material.color.setHex(0xffaa00);
                    material.opacity = 1.0;
                }
            }
        });
    }

    private clearHighlights(): void {
        this.edges.forEach(edge => {
            if (edge.line) {
                const material = edge.line.material as THREE.LineBasicMaterial;
                material.color.copy(edge.color);
                material.opacity = 0.6;
            }
        });
    }

    private handleResize(): void {
        const container = this.renderer.domElement.parentElement!;
        const width = container.clientWidth;
        const height = container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    private animate = (): void => {
        if (!this.animationState.playing) return;

        const currentTime = performance.now() * 0.001;
        this.animationState.deltaTime = (currentTime - this.animationState.time) * this.animationState.speed;
        this.animationState.time = currentTime;

        // Run physics simulation
        for (let i = 0; i < this.config.layoutSteps; i++) {
            this.updateForces();
            this.updatePositions();
        }

        // Update particle system
        if (this.particleSystem) {
            this.particleSystem.rotation.x += 0.001;
            this.particleSystem.rotation.y += 0.002;
        }

        this.renderer.render(this.scene, this.camera);
        this.animationId = requestAnimationFrame(this.animate);
    };

    public startAnimation(): void {
        this.animationState.playing = true;
        if (!this.animationId) {
            this.animate();
        }
    }

    public stopAnimation(): void {
        this.animationState.playing = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    public updateNodeActivation(nodeId: string, activation: number): void {
        const node = this.nodes.get(nodeId);
        if (node) {
            node.activation = Math.max(0, Math.min(1, activation));
        }
    }

    public animateActivationPropagation(path: string[]): void {
        path.forEach((nodeId, index) => {
            setTimeout(() => {
                this.updateNodeActivation(nodeId, 1.0);
                setTimeout(() => {
                    this.updateNodeActivation(nodeId, 0.0);
                }, 500);
            }, index * 200);
        });
    }

    public getNode(nodeId: string): GraphNode | undefined {
        return this.nodes.get(nodeId);
    }

    public getEdge(edgeId: string): GraphEdge | undefined {
        return this.edges.get(edgeId);
    }

    public removeNode(nodeId: string): void {
        const node = this.nodes.get(nodeId);
        if (node) {
            if (node.mesh) this.scene.remove(node.mesh);
            if (node.label) this.scene.remove(node.label);
            this.nodes.delete(nodeId);
        }
    }

    public removeEdge(edgeId: string): void {
        const edge = this.edges.get(edgeId);
        if (edge) {
            if (edge.line) this.scene.remove(edge.line);
            if (edge.particles) this.scene.remove(edge.particles);
            this.edges.delete(edgeId);
        }
    }

    public clear(): void {
        this.nodes.forEach((_, nodeId) => this.removeNode(nodeId));
        this.edges.forEach((_, edgeId) => this.removeEdge(edgeId));
    }

    public dispose(): void {
        this.stopAnimation();
        this.clear();
        this.renderer.dispose();
    }
}