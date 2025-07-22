# Phase 4: Data Flow Visualization

## Overview

Phase 4 implements an advanced animated graph visualization system for LLMKG data flows, providing real-time insights into how data moves through the cognitive patterns, neural operations, and knowledge graph structures. This phase creates beautiful, performant visualizations that showcase the unique brain-inspired architecture of LLMKG.

## Core Components

### 1. Animated Graph Visualization System

#### Architecture Overview

```javascript
// Core visualization engine using D3.js and Three.js
class LLMKGDataFlowVisualizer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            width: options.width || window.innerWidth,
            height: options.height || window.innerHeight,
            animationSpeed: options.animationSpeed || 1.0,
            particleCount: options.particleCount || 1000,
            enableWebGL: options.enableWebGL !== false,
            theme: options.theme || 'neural',
            ...options
        };
        
        this.scene = null;
        this.renderer = null;
        this.camera = null;
        this.controls = null;
        this.dataFlows = new Map();
        this.activePatterns = new Set();
        this.sdrVisualizer = null;
        
        this.init();
    }
    
    init() {
        if (this.options.enableWebGL) {
            this.initThreeJS();
        }
        this.initD3();
        this.initEventHandlers();
        this.startAnimationLoop();
    }
    
    initThreeJS() {
        // Three.js setup for 3D visualization
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75, 
            this.options.width / this.options.height, 
            0.1, 
            10000
        );
        
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true 
        });
        this.renderer.setSize(this.options.width, this.options.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        
        // Add fog for depth perception
        this.scene.fog = new THREE.FogExp2(0x000000, 0.0008);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 0.5).normalize();
        this.scene.add(directionalLight);
        
        // Camera controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        this.container.appendChild(this.renderer.domElement);
        this.camera.position.z = 1000;
    }
    
    initD3() {
        // D3.js overlay for 2D elements
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height)
            .style('position', 'absolute')
            .style('top', 0)
            .style('left', 0)
            .style('pointer-events', 'none');
        
        // Define gradients and filters
        const defs = this.svg.append('defs');
        
        // Neural glow filter
        const filter = defs.append('filter')
            .attr('id', 'neural-glow');
        
        filter.append('feGaussianBlur')
            .attr('stdDeviation', '4')
            .attr('result', 'coloredBlur');
            
        const feMerge = filter.append('feMerge');
        feMerge.append('feMergeNode').attr('in', 'coloredBlur');
        feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
    }
}
```

### 2. MCP Request Tracing

```javascript
class MCPRequestTracer {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.activeRequests = new Map();
        this.requestPaths = new Map();
        this.particleSystem = new MCPParticleSystem(visualizer.scene);
    }
    
    traceRequest(requestId, requestData) {
        const trace = {
            id: requestId,
            startTime: Date.now(),
            path: [],
            nodes: [],
            data: requestData,
            particles: []
        };
        
        this.activeRequests.set(requestId, trace);
        this.visualizeRequestStart(trace);
        
        return {
            addNode: (nodeData) => this.addTraceNode(requestId, nodeData),
            complete: () => this.completeTrace(requestId),
            error: (error) => this.errorTrace(requestId, error)
        };
    }
    
    visualizeRequestStart(trace) {
        // Create visual representation of request initiation
        const startNode = this.createRequestNode(trace);
        
        // Animate request particle
        const particle = this.particleSystem.createParticle({
            color: this.getRequestColor(trace.data.type),
            size: 5,
            trail: true,
            glow: true
        });
        
        trace.particles.push(particle);
        
        // Animate from entry point
        gsap.from(particle.position, {
            duration: 0.5,
            x: -1000,
            ease: "power2.out",
            onUpdate: () => {
                this.particleSystem.updateTrail(particle);
            }
        });
    }
    
    addTraceNode(requestId, nodeData) {
        const trace = this.activeRequests.get(requestId);
        if (!trace) return;
        
        const node = {
            timestamp: Date.now(),
            type: nodeData.type,
            component: nodeData.component,
            data: nodeData.data,
            position: this.calculateNodePosition(trace.nodes.length)
        };
        
        trace.nodes.push(node);
        trace.path.push(node.position);
        
        // Visualize node processing
        this.visualizeNode(node, trace);
        
        // Animate particle movement
        if (trace.particles.length > 0) {
            const particle = trace.particles[0];
            gsap.to(particle.position, {
                duration: 0.3,
                x: node.position.x,
                y: node.position.y,
                z: node.position.z,
                ease: "power1.inOut",
                onUpdate: () => {
                    this.particleSystem.updateTrail(particle);
                }
            });
        }
    }
    
    visualizeNode(node, trace) {
        // Create 3D node representation
        const geometry = this.getNodeGeometry(node.type);
        const material = new THREE.MeshPhongMaterial({
            color: this.getNodeColor(node.type),
            emissive: this.getNodeColor(node.type),
            emissiveIntensity: 0.2,
            transparent: true,
            opacity: 0.8
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(node.position);
        
        // Add to scene with animation
        mesh.scale.set(0, 0, 0);
        this.visualizer.scene.add(mesh);
        
        gsap.to(mesh.scale, {
            duration: 0.3,
            x: 1,
            y: 1,
            z: 1,
            ease: "back.out(1.7)"
        });
        
        // Add label
        this.addNodeLabel(node, mesh);
        
        // Store reference
        node.mesh = mesh;
    }
    
    getNodeGeometry(type) {
        switch (type) {
            case 'cognitive':
                return new THREE.OctahedronGeometry(20);
            case 'neural':
                return new THREE.IcosahedronGeometry(15);
            case 'storage':
                return new THREE.BoxGeometry(25, 25, 25);
            case 'query':
                return new THREE.CylinderGeometry(15, 15, 30);
            default:
                return new THREE.SphereGeometry(15);
        }
    }
    
    getNodeColor(type) {
        const colors = {
            cognitive: 0x00ff88,
            neural: 0xff0088,
            storage: 0x0088ff,
            query: 0xffaa00,
            memory: 0x8800ff
        };
        return colors[type] || 0xffffff;
    }
}
```

### 3. Cognitive Pattern Activation Visualization

```javascript
class CognitivePatternVisualizer {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.patterns = new Map();
        this.activationWaves = [];
        this.neuralNetwork = new NeuralNetworkVisualizer(visualizer.scene);
    }
    
    visualizePattern(patternType, activationData) {
        const pattern = this.getOrCreatePattern(patternType);
        
        // Update activation levels
        pattern.activation = activationData.level;
        pattern.connections = activationData.connections;
        
        // Create activation wave effect
        this.createActivationWave(pattern, activationData);
        
        // Update neural network connections
        this.updateNeuralConnections(pattern, activationData);
        
        // Animate pattern-specific effects
        this.animatePatternEffects(patternType, pattern, activationData);
    }
    
    createActivationWave(pattern, activationData) {
        const wave = new ActivationWave({
            center: pattern.position,
            radius: activationData.level * 100,
            color: pattern.color,
            intensity: activationData.level,
            speed: 2.0
        });
        
        this.activationWaves.push(wave);
        
        // Animate wave expansion
        wave.animate({
            duration: 2,
            onUpdate: (progress) => {
                wave.updateShader(progress);
                
                // Affect nearby nodes
                this.propagateActivation(wave, progress);
            },
            onComplete: () => {
                this.activationWaves = this.activationWaves.filter(w => w !== wave);
                wave.dispose();
            }
        });
    }
    
    animatePatternEffects(patternType, pattern, activationData) {
        switch (patternType) {
            case 'divergent':
                this.animateDivergentThinking(pattern, activationData);
                break;
            case 'convergent':
                this.animateConvergentThinking(pattern, activationData);
                break;
            case 'lateral':
                this.animateLateralThinking(pattern, activationData);
                break;
            case 'critical':
                this.animateCriticalThinking(pattern, activationData);
                break;
            case 'inhibitory':
                this.animateInhibitoryMechanism(pattern, activationData);
                break;
        }
    }
    
    animateDivergentThinking(pattern, data) {
        // Create branching particle effects
        const particleCount = Math.floor(data.level * 50);
        const particles = [];
        
        for (let i = 0; i < particleCount; i++) {
            const particle = new THREE.Sprite(
                new THREE.SpriteMaterial({
                    map: this.createParticleTexture(),
                    color: 0x00ff00,
                    blending: THREE.AdditiveBlending,
                    transparent: true
                })
            );
            
            particle.position.copy(pattern.position);
            particle.scale.set(10, 10, 1);
            
            // Random divergent direction
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;
            particle.velocity = new THREE.Vector3(
                Math.sin(phi) * Math.cos(theta),
                Math.sin(phi) * Math.sin(theta),
                Math.cos(phi)
            ).multiplyScalar(2 + Math.random() * 3);
            
            particles.push(particle);
            this.visualizer.scene.add(particle);
        }
        
        // Animate particles
        const animate = () => {
            particles.forEach((particle, index) => {
                particle.position.add(particle.velocity);
                particle.material.opacity *= 0.98;
                
                if (particle.material.opacity < 0.01) {
                    this.visualizer.scene.remove(particle);
                    particles.splice(index, 1);
                }
            });
            
            if (particles.length > 0) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    animateInhibitoryMechanism(pattern, data) {
        // Create inhibition field effect
        const fieldGeometry = new THREE.SphereGeometry(
            data.radius || 100,
            32,
            32
        );
        
        const fieldMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                inhibitionStrength: { value: data.level },
                color: { value: new THREE.Color(0xff0000) }
            },
            vertexShader: `
                varying vec3 vPosition;
                varying vec3 vNormal;
                
                void main() {
                    vPosition = position;
                    vNormal = normal;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform float inhibitionStrength;
                uniform vec3 color;
                
                varying vec3 vPosition;
                varying vec3 vNormal;
                
                void main() {
                    float pulse = sin(time * 2.0) * 0.5 + 0.5;
                    float alpha = inhibitionStrength * pulse * 0.3;
                    
                    // Create ripple effect
                    float dist = length(vPosition);
                    float ripple = sin(dist * 0.1 - time * 3.0) * 0.5 + 0.5;
                    
                    vec3 finalColor = mix(color, vec3(1.0, 0.0, 0.0), ripple);
                    
                    gl_FragColor = vec4(finalColor, alpha);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending
        });
        
        const field = new THREE.Mesh(fieldGeometry, fieldMaterial);
        field.position.copy(pattern.position);
        this.visualizer.scene.add(field);
        
        // Animate the inhibition field
        const animateField = () => {
            fieldMaterial.uniforms.time.value += 0.01;
            fieldMaterial.uniforms.inhibitionStrength.value = 
                data.level * (0.8 + Math.sin(Date.now() * 0.001) * 0.2);
            
            requestAnimationFrame(animateField);
        };
        
        animateField();
    }
}
```

### 4. Memory and Storage Operation Visualization

```javascript
class MemoryOperationVisualizer {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.memoryBlocks = new Map();
        this.operationQueue = [];
        this.sdrVisualizer = new SDRVisualizer(visualizer);
    }
    
    visualizeMemoryOperation(operation) {
        const op = {
            id: operation.id,
            type: operation.type, // read, write, update, delete
            address: operation.address,
            size: operation.size,
            data: operation.data,
            timestamp: Date.now()
        };
        
        this.operationQueue.push(op);
        
        switch (op.type) {
            case 'read':
                this.visualizeRead(op);
                break;
            case 'write':
                this.visualizeWrite(op);
                break;
            case 'sdr_encode':
                this.visualizeSDREncoding(op);
                break;
            case 'sdr_decode':
                this.visualizeSDRDecoding(op);
                break;
        }
    }
    
    visualizeSDREncoding(operation) {
        // Visualize Sparse Distributed Representation encoding
        const sdrData = operation.data.sdr;
        const bitArray = sdrData.bits;
        const dimensions = sdrData.dimensions || [1024, 1024];
        
        // Create 3D representation of SDR
        const sdrGeometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const sizes = [];
        
        // Convert bit positions to 3D coordinates
        bitArray.forEach((bitIndex) => {
            const x = (bitIndex % dimensions[0]) - dimensions[0] / 2;
            const y = Math.floor(bitIndex / dimensions[0]) % dimensions[1] - dimensions[1] / 2;
            const z = Math.floor(bitIndex / (dimensions[0] * dimensions[1])) * 10;
            
            positions.push(x, y, z);
            
            // Color based on activation pattern
            const hue = (bitIndex / (dimensions[0] * dimensions[1])) * 360;
            const color = new THREE.Color().setHSL(hue / 360, 1.0, 0.5);
            colors.push(color.r, color.g, color.b);
            
            // Size based on bit significance
            sizes.push(5 + Math.random() * 10);
        });
        
        sdrGeometry.setAttribute('position', 
            new THREE.Float32BufferAttribute(positions, 3));
        sdrGeometry.setAttribute('color', 
            new THREE.Float32BufferAttribute(colors, 3));
        sdrGeometry.setAttribute('size', 
            new THREE.Float32BufferAttribute(sizes, 1));
        
        // Custom shader for SDR visualization
        const sdrMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                pixelRatio: { value: window.devicePixelRatio }
            },
            vertexShader: `
                attribute float size;
                attribute vec3 color;
                varying vec3 vColor;
                
                void main() {
                    vColor = color;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * (300.0 / -mvPosition.z) * pixelRatio;
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform float time;
                varying vec3 vColor;
                
                void main() {
                    vec2 center = gl_PointCoord - vec2(0.5);
                    float dist = length(center);
                    
                    if (dist > 0.5) {
                        discard;
                    }
                    
                    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
                    float pulse = sin(time * 3.0) * 0.2 + 0.8;
                    
                    gl_FragColor = vec4(vColor * pulse, alpha);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        
        const sdrPoints = new THREE.Points(sdrGeometry, sdrMaterial);
        sdrPoints.rotation.x = Math.PI / 4;
        
        // Animate SDR formation
        sdrPoints.scale.set(0, 0, 0);
        this.visualizer.scene.add(sdrPoints);
        
        gsap.to(sdrPoints.scale, {
            duration: 1,
            x: 1,
            y: 1,
            z: 1,
            ease: "elastic.out(1, 0.5)",
            onUpdate: () => {
                sdrMaterial.uniforms.time.value += 0.05;
            }
        });
        
        // Store reference
        this.memoryBlocks.set(operation.address, {
            mesh: sdrPoints,
            data: sdrData,
            type: 'sdr'
        });
    }
    
    visualizeWrite(operation) {
        // Create memory block visualization
        const blockSize = Math.cbrt(operation.size) * 10;
        const geometry = new THREE.BoxGeometry(blockSize, blockSize, blockSize);
        
        const material = new THREE.MeshPhongMaterial({
            color: 0x00ff00,
            emissive: 0x00ff00,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.8
        });
        
        const block = new THREE.Mesh(geometry, material);
        block.position.set(
            (operation.address % 100) * 30 - 1500,
            Math.floor(operation.address / 100) * 30 - 500,
            0
        );
        
        // Write animation
        const writeParticles = this.createWriteParticles(block.position, operation.size);
        
        // Converge particles to form block
        writeParticles.forEach((particle, index) => {
            gsap.to(particle.position, {
                duration: 0.5 + index * 0.01,
                x: block.position.x + (Math.random() - 0.5) * blockSize,
                y: block.position.y + (Math.random() - 0.5) * blockSize,
                z: block.position.z + (Math.random() - 0.5) * blockSize,
                ease: "power2.in",
                onComplete: () => {
                    this.visualizer.scene.remove(particle);
                    if (index === writeParticles.length - 1) {
                        this.visualizer.scene.add(block);
                        gsap.from(block.scale, {
                            duration: 0.3,
                            x: 0,
                            y: 0,
                            z: 0,
                            ease: "back.out(1.7)"
                        });
                    }
                }
            });
        });
        
        this.memoryBlocks.set(operation.address, {
            mesh: block,
            data: operation.data,
            type: 'standard'
        });
    }
}
```

### 5. Real-time Knowledge Graph Query Animation

```javascript
class KnowledgeGraphQueryVisualizer {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.graphNodes = new Map();
        this.graphEdges = new Map();
        this.queryPaths = [];
        this.forceSimulation = null;
        
        this.initForceLayout();
    }
    
    initForceLayout() {
        // D3 force simulation for graph layout
        this.forceSimulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(0, 0))
            .force("collision", d3.forceCollide().radius(30));
    }
    
    visualizeQuery(query) {
        const queryVis = {
            id: query.id,
            type: query.type,
            startNode: query.startNode,
            targetPattern: query.pattern,
            timestamp: Date.now(),
            paths: [],
            activePath: null
        };
        
        // Start query animation
        this.animateQueryStart(queryVis);
        
        // Return query controller
        return {
            addNode: (node) => this.addQueryNode(queryVis, node),
            addEdge: (edge) => this.addQueryEdge(queryVis, edge),
            highlightPath: (path) => this.highlightQueryPath(queryVis, path),
            complete: (results) => this.completeQuery(queryVis, results)
        };
    }
    
    animateQueryStart(query) {
        // Create query beacon at start node
        const startNode = this.graphNodes.get(query.startNode);
        if (!startNode) return;
        
        const beacon = this.createQueryBeacon(query);
        beacon.position.copy(startNode.position);
        
        // Pulse effect
        const pulseAnimation = () => {
            beacon.material.uniforms.time.value += 0.05;
            beacon.material.uniforms.radius.value = 20 + Math.sin(Date.now() * 0.003) * 10;
            
            if (query.active) {
                requestAnimationFrame(pulseAnimation);
            }
        };
        
        query.beacon = beacon;
        query.active = true;
        pulseAnimation();
    }
    
    createQueryBeacon(query) {
        const geometry = new THREE.SphereGeometry(20, 32, 32);
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                radius: { value: 20 },
                queryColor: { value: new THREE.Color(this.getQueryColor(query.type)) }
            },
            vertexShader: `
                varying vec3 vPosition;
                uniform float time;
                uniform float radius;
                
                void main() {
                    vPosition = position;
                    vec3 pos = position;
                    
                    // Wave distortion
                    float wave = sin(position.x * 0.1 + time) * 
                                 sin(position.y * 0.1 + time) * 
                                 sin(position.z * 0.1 + time) * 5.0;
                    
                    pos *= 1.0 + wave * 0.01;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 queryColor;
                uniform float time;
                varying vec3 vPosition;
                
                void main() {
                    float dist = length(vPosition);
                    float glow = 1.0 / (dist * 0.1 + 1.0);
                    
                    vec3 color = queryColor * glow;
                    float alpha = glow * 0.8;
                    
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending
        });
        
        const beacon = new THREE.Mesh(geometry, material);
        this.visualizer.scene.add(beacon);
        
        return beacon;
    }
    
    highlightQueryPath(query, path) {
        // Create animated path visualization
        const pathPoints = path.map(nodeId => {
            const node = this.graphNodes.get(nodeId);
            return node ? node.position.clone() : null;
        }).filter(p => p !== null);
        
        if (pathPoints.length < 2) return;
        
        // Create path curve
        const curve = new THREE.CatmullRomCurve3(pathPoints);
        const points = curve.getPoints(100);
        
        // Animated path line
        const pathGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const pathMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                pathLength: { value: points.length },
                queryColor: { value: new THREE.Color(this.getQueryColor(query.type)) }
            },
            vertexShader: `
                attribute float lineDistance;
                varying float vLineDistance;
                uniform float time;
                
                void main() {
                    vLineDistance = lineDistance;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    
                    // Add wave motion
                    mvPosition.xyz += sin(lineDistance * 0.1 + time) * 2.0;
                    
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform float pathLength;
                uniform vec3 queryColor;
                varying float vLineDistance;
                
                void main() {
                    // Animated gradient along path
                    float progress = mod(vLineDistance + time * 50.0, pathLength) / pathLength;
                    float alpha = 1.0 - progress;
                    
                    // Pulse effect
                    alpha *= 0.5 + sin(time * 3.0) * 0.5;
                    
                    gl_FragColor = vec4(queryColor, alpha);
                }
            `,
            transparent: true,
            linewidth: 3,
            blending: THREE.AdditiveBlending
        });
        
        const pathLine = new THREE.Line(pathGeometry, pathMaterial);
        this.visualizer.scene.add(pathLine);
        
        query.activePath = pathLine;
        
        // Animate path
        const animatePath = () => {
            pathMaterial.uniforms.time.value += 0.1;
            
            if (query.active) {
                requestAnimationFrame(animatePath);
            }
        };
        
        animatePath();
    }
}
```

### 6. Entity and Relationship Flow Tracking

```javascript
class EntityRelationshipFlowTracker {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.entities = new Map();
        this.relationships = new Map();
        this.flowParticles = [];
        this.flowField = new FlowField(visualizer.scene);
    }
    
    trackEntityCreation(entity) {
        const entityVis = this.createEntityVisualization(entity);
        this.entities.set(entity.id, entityVis);
        
        // Animate entity emergence
        this.animateEntityEmergence(entityVis);
        
        // Update flow field
        this.flowField.addSource(entityVis.position, entity.strength || 1.0);
    }
    
    trackRelationshipFormation(relationship) {
        const source = this.entities.get(relationship.sourceId);
        const target = this.entities.get(relationship.targetId);
        
        if (!source || !target) return;
        
        const relVis = this.createRelationshipVisualization(
            source, 
            target, 
            relationship
        );
        
        this.relationships.set(relationship.id, relVis);
        
        // Create flow particles
        this.createRelationshipFlow(relVis, relationship);
    }
    
    createEntityVisualization(entity) {
        // Entity type determines visual representation
        const geometryConfig = this.getEntityGeometry(entity.type);
        const geometry = geometryConfig.create();
        
        // Material with custom shaders for entity rendering
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                entityColor: { value: new THREE.Color(entity.color || 0x00ffff) },
                salience: { value: entity.salience || 1.0 },
                activation: { value: 0 }
            },
            vertexShader: `
                varying vec3 vNormal;
                varying vec3 vPosition;
                uniform float time;
                uniform float activation;
                
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    vPosition = position;
                    
                    // Breathing effect
                    vec3 pos = position * (1.0 + sin(time * 2.0) * 0.05 * activation);
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 entityColor;
                uniform float salience;
                uniform float activation;
                uniform float time;
                
                varying vec3 vNormal;
                varying vec3 vPosition;
                
                void main() {
                    // Fresnel effect for rim lighting
                    vec3 viewDirection = normalize(cameraPosition - vPosition);
                    float fresnel = pow(1.0 - dot(viewDirection, vNormal), 2.0);
                    
                    // Core color with salience-based intensity
                    vec3 color = entityColor * salience;
                    
                    // Add rim glow
                    color += entityColor * fresnel * activation;
                    
                    // Pulse effect
                    float pulse = sin(time * 3.0) * 0.2 + 0.8;
                    
                    gl_FragColor = vec4(color * pulse, 0.8 + fresnel * 0.2);
                }
            `,
            transparent: true
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(
            Math.random() * 1000 - 500,
            Math.random() * 1000 - 500,
            Math.random() * 500 - 250
        );
        
        this.visualizer.scene.add(mesh);
        
        return {
            id: entity.id,
            mesh: mesh,
            material: material,
            entity: entity,
            position: mesh.position,
            connections: new Set()
        };
    }
    
    createRelationshipFlow(relVis, relationship) {
        const flowConfig = {
            source: relVis.source.position,
            target: relVis.target.position,
            strength: relationship.strength || 1.0,
            bidirectional: relationship.bidirectional || false,
            particleCount: Math.floor(relationship.strength * 10) || 5,
            color: relationship.color || 0xffffff
        };
        
        // Create flowing particles along relationship
        const particles = [];
        
        for (let i = 0; i < flowConfig.particleCount; i++) {
            const particle = new FlowParticle({
                path: relVis.curve,
                speed: 0.5 + Math.random() * 0.5,
                size: 3 + relationship.strength * 2,
                color: flowConfig.color,
                offset: i / flowConfig.particleCount
            });
            
            particles.push(particle);
            this.flowParticles.push(particle);
            this.visualizer.scene.add(particle.sprite);
        }
        
        relVis.particles = particles;
        
        // Start flow animation
        this.animateFlow(relVis);
    }
    
    animateFlow(relVis) {
        const animate = () => {
            relVis.particles.forEach(particle => {
                particle.update();
                
                // Emit trail particles
                if (Math.random() < 0.1) {
                    this.emitTrailParticle(particle);
                }
            });
            
            if (relVis.active) {
                requestAnimationFrame(animate);
            }
        };
        
        relVis.active = true;
        animate();
    }
}
```

### 7. Performance Impact Visualization

```javascript
class PerformanceImpactVisualizer {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.metrics = {
            fps: [],
            memoryUsage: [],
            queryLatency: [],
            throughput: []
        };
        this.charts = {};
        this.heatmap = null;
        
        this.initPerformanceDisplay();
    }
    
    initPerformanceDisplay() {
        // Create performance overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'performance-overlay';
        this.overlay.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            width: 300px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #00ff00;
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            color: #00ff00;
            z-index: 1000;
        `;
        
        this.visualizer.container.appendChild(this.overlay);
        
        // Initialize charts
        this.initFPSChart();
        this.initLatencyHeatmap();
        this.initThroughputGauge();
    }
    
    initLatencyHeatmap() {
        // Create 3D heatmap for system latency
        const heatmapSize = 50;
        const geometry = new THREE.PlaneGeometry(500, 500, heatmapSize, heatmapSize);
        
        // Custom shader for heatmap
        const material = new THREE.ShaderMaterial({
            uniforms: {
                latencyData: { 
                    value: new Float32Array(heatmapSize * heatmapSize).fill(0) 
                },
                maxLatency: { value: 100 },
                time: { value: 0 }
            },
            vertexShader: `
                varying vec2 vUv;
                varying float vLatency;
                uniform float latencyData[${heatmapSize * heatmapSize}];
                
                void main() {
                    vUv = uv;
                    
                    // Get latency for this vertex
                    int index = int(uv.x * ${heatmapSize}.0) + 
                               int(uv.y * ${heatmapSize}.0) * ${heatmapSize};
                    vLatency = latencyData[index];
                    
                    // Displace vertex based on latency
                    vec3 pos = position;
                    pos.z = vLatency * 2.0;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                varying vec2 vUv;
                varying float vLatency;
                uniform float maxLatency;
                uniform float time;
                
                vec3 getHeatmapColor(float value) {
                    // Blue -> Green -> Yellow -> Red
                    vec3 blue = vec3(0.0, 0.0, 1.0);
                    vec3 green = vec3(0.0, 1.0, 0.0);
                    vec3 yellow = vec3(1.0, 1.0, 0.0);
                    vec3 red = vec3(1.0, 0.0, 0.0);
                    
                    float t = value / maxLatency;
                    
                    if (t < 0.33) {
                        return mix(blue, green, t * 3.0);
                    } else if (t < 0.66) {
                        return mix(green, yellow, (t - 0.33) * 3.0);
                    } else {
                        return mix(yellow, red, (t - 0.66) * 3.0);
                    }
                }
                
                void main() {
                    vec3 color = getHeatmapColor(vLatency);
                    
                    // Add pulse for high latency areas
                    if (vLatency > maxLatency * 0.8) {
                        float pulse = sin(time * 5.0) * 0.5 + 0.5;
                        color = mix(color, vec3(1.0, 0.0, 0.0), pulse);
                    }
                    
                    gl_FragColor = vec4(color, 0.9);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide
        });
        
        this.heatmap = new THREE.Mesh(geometry, material);
        this.heatmap.rotation.x = -Math.PI / 2;
        this.heatmap.position.y = -200;
        this.visualizer.scene.add(this.heatmap);
    }
    
    updateMetrics(metrics) {
        // Update performance metrics
        Object.keys(metrics).forEach(key => {
            if (this.metrics[key]) {
                this.metrics[key].push({
                    value: metrics[key],
                    timestamp: Date.now()
                });
                
                // Keep only recent data
                if (this.metrics[key].length > 100) {
                    this.metrics[key].shift();
                }
            }
        });
        
        // Update visualizations
        this.updateFPSChart();
        this.updateLatencyHeatmap(metrics.latencyMap);
        this.updateThroughputGauge(metrics.throughput);
        
        // Update overlay text
        this.updateOverlayText(metrics);
    }
    
    updateLatencyHeatmap(latencyMap) {
        if (!this.heatmap || !latencyMap) return;
        
        const material = this.heatmap.material;
        const latencyData = material.uniforms.latencyData.value;
        
        // Update latency data
        let maxLatency = 0;
        latencyMap.forEach((latency, index) => {
            latencyData[index] = latency;
            maxLatency = Math.max(maxLatency, latency);
        });
        
        material.uniforms.maxLatency.value = maxLatency;
        material.uniforms.time.value += 0.016;
        
        // Update geometry if needed
        const geometry = this.heatmap.geometry;
        const positions = geometry.attributes.position.array;
        
        for (let i = 0; i < positions.length; i += 3) {
            const x = Math.floor((i / 3) % 51);
            const y = Math.floor((i / 3) / 51);
            const index = y * 50 + x;
            
            if (index < latencyData.length) {
                positions[i + 2] = latencyData[index] * 2;
            }
        }
        
        geometry.attributes.position.needsUpdate = true;
    }
}
```

### 8. Interactive Controls and Filtering

```javascript
class InteractiveControlPanel {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.filters = {
            patternTypes: new Set(['all']),
            operationTypes: new Set(['all']),
            minActivation: 0,
            timeRange: { start: null, end: null }
        };
        
        this.initControlPanel();
        this.initKeyboardControls();
        this.initMouseControls();
    }
    
    initControlPanel() {
        // Create control panel UI
        this.panel = document.createElement('div');
        this.panel.className = 'control-panel';
        this.panel.innerHTML = `
            <div class="control-header">
                <h3>LLMKG Data Flow Controls</h3>
                <button class="minimize-btn">_</button>
            </div>
            
            <div class="control-body">
                <!-- Pattern Type Filters -->
                <div class="control-section">
                    <h4>Cognitive Patterns</h4>
                    <div class="pattern-filters">
                        <label><input type="checkbox" value="all" checked> All</label>
                        <label><input type="checkbox" value="divergent"> Divergent</label>
                        <label><input type="checkbox" value="convergent"> Convergent</label>
                        <label><input type="checkbox" value="lateral"> Lateral</label>
                        <label><input type="checkbox" value="critical"> Critical</label>
                        <label><input type="checkbox" value="inhibitory"> Inhibitory</label>
                    </div>
                </div>
                
                <!-- Operation Type Filters -->
                <div class="control-section">
                    <h4>Operation Types</h4>
                    <div class="operation-filters">
                        <label><input type="checkbox" value="all" checked> All</label>
                        <label><input type="checkbox" value="query"> Queries</label>
                        <label><input type="checkbox" value="write"> Writes</label>
                        <label><input type="checkbox" value="sdr"> SDR Operations</label>
                        <label><input type="checkbox" value="memory"> Memory Ops</label>
                    </div>
                </div>
                
                <!-- Activation Threshold -->
                <div class="control-section">
                    <h4>Activation Threshold</h4>
                    <input type="range" id="activation-threshold" 
                           min="0" max="100" value="0" step="1">
                    <span id="activation-value">0%</span>
                </div>
                
                <!-- Time Range -->
                <div class="control-section">
                    <h4>Time Range</h4>
                    <button id="time-live">Live</button>
                    <button id="time-1min">Last 1 min</button>
                    <button id="time-5min">Last 5 min</button>
                    <button id="time-custom">Custom</button>
                </div>
                
                <!-- Visualization Controls -->
                <div class="control-section">
                    <h4>Visualization</h4>
                    <label><input type="checkbox" id="show-particles" checked> 
                        Show Particles</label>
                    <label><input type="checkbox" id="show-trails" checked> 
                        Show Trails</label>
                    <label><input type="checkbox" id="show-labels" checked> 
                        Show Labels</label>
                    <label><input type="checkbox" id="show-performance"> 
                        Show Performance</label>
                </div>
                
                <!-- Animation Controls -->
                <div class="control-section">
                    <h4>Animation Speed</h4>
                    <input type="range" id="animation-speed" 
                           min="0" max="200" value="100" step="10">
                    <span id="speed-value">1.0x</span>
                </div>
                
                <!-- Actions -->
                <div class="control-section actions">
                    <button id="pause-btn">Pause</button>
                    <button id="reset-view">Reset View</button>
                    <button id="export-data">Export Data</button>
                    <button id="screenshot">Screenshot</button>
                </div>
            </div>
        `;
        
        // Apply styles
        this.applyPanelStyles();
        
        // Add to container
        this.visualizer.container.appendChild(this.panel);
        
        // Bind event handlers
        this.bindControlEvents();
    }
    
    applyPanelStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .control-panel {
                position: absolute;
                left: 10px;
                top: 10px;
                width: 250px;
                background: rgba(0, 0, 0, 0.9);
                border: 1px solid #00ff88;
                border-radius: 5px;
                color: #00ff88;
                font-family: 'Roboto Mono', monospace;
                font-size: 12px;
                z-index: 1000;
                box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
            }
            
            .control-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                border-bottom: 1px solid #00ff88;
                background: rgba(0, 255, 136, 0.1);
            }
            
            .control-header h3 {
                margin: 0;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .minimize-btn {
                background: none;
                border: 1px solid #00ff88;
                color: #00ff88;
                cursor: pointer;
                padding: 2px 8px;
                border-radius: 3px;
                transition: all 0.3s;
            }
            
            .minimize-btn:hover {
                background: #00ff88;
                color: #000;
            }
            
            .control-body {
                padding: 10px;
                max-height: 600px;
                overflow-y: auto;
            }
            
            .control-section {
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid rgba(0, 255, 136, 0.3);
            }
            
            .control-section:last-child {
                border-bottom: none;
            }
            
            .control-section h4 {
                margin: 0 0 8px 0;
                font-size: 12px;
                color: #00ff88;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .control-section label {
                display: block;
                margin: 5px 0;
                cursor: pointer;
                transition: color 0.3s;
            }
            
            .control-section label:hover {
                color: #ffffff;
            }
            
            .control-section input[type="checkbox"] {
                margin-right: 8px;
                cursor: pointer;
            }
            
            .control-section input[type="range"] {
                width: 100%;
                margin: 5px 0;
            }
            
            .control-section button {
                background: rgba(0, 255, 136, 0.2);
                border: 1px solid #00ff88;
                color: #00ff88;
                padding: 5px 10px;
                margin: 2px;
                cursor: pointer;
                border-radius: 3px;
                transition: all 0.3s;
                font-size: 11px;
            }
            
            .control-section button:hover {
                background: #00ff88;
                color: #000;
            }
            
            .actions {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }
            
            .actions button {
                flex: 1 1 45%;
            }
            
            /* Custom scrollbar */
            .control-body::-webkit-scrollbar {
                width: 8px;
            }
            
            .control-body::-webkit-scrollbar-track {
                background: rgba(0, 255, 136, 0.1);
            }
            
            .control-body::-webkit-scrollbar-thumb {
                background: rgba(0, 255, 136, 0.5);
                border-radius: 4px;
            }
            
            .control-body::-webkit-scrollbar-thumb:hover {
                background: rgba(0, 255, 136, 0.8);
            }
        `;
        
        document.head.appendChild(style);
    }
    
    bindControlEvents() {
        // Pattern filters
        this.panel.querySelectorAll('.pattern-filters input').forEach(input => {
            input.addEventListener('change', (e) => {
                this.updatePatternFilter(e.target.value, e.target.checked);
            });
        });
        
        // Operation filters
        this.panel.querySelectorAll('.operation-filters input').forEach(input => {
            input.addEventListener('change', (e) => {
                this.updateOperationFilter(e.target.value, e.target.checked);
            });
        });
        
        // Activation threshold
        const activationSlider = this.panel.querySelector('#activation-threshold');
        const activationValue = this.panel.querySelector('#activation-value');
        
        activationSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            activationValue.textContent = `${value}%`;
            this.filters.minActivation = value / 100;
            this.applyFilters();
        });
        
        // Animation speed
        const speedSlider = this.panel.querySelector('#animation-speed');
        const speedValue = this.panel.querySelector('#speed-value');
        
        speedSlider.addEventListener('input', (e) => {
            const value = e.target.value / 100;
            speedValue.textContent = `${value.toFixed(1)}x`;
            this.visualizer.options.animationSpeed = value;
        });
        
        // Action buttons
        this.panel.querySelector('#pause-btn').addEventListener('click', () => {
            this.togglePause();
        });
        
        this.panel.querySelector('#reset-view').addEventListener('click', () => {
            this.resetView();
        });
        
        this.panel.querySelector('#export-data').addEventListener('click', () => {
            this.exportVisualizationData();
        });
        
        this.panel.querySelector('#screenshot').addEventListener('click', () => {
            this.takeScreenshot();
        });
    }
    
    initKeyboardControls() {
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    this.togglePause();
                    break;
                case 'r':
                    this.resetView();
                    break;
                case 'f':
                    this.toggleFullscreen();
                    break;
                case 'h':
                    this.toggleControlPanel();
                    break;
                case '+':
                case '=':
                    this.zoomIn();
                    break;
                case '-':
                case '_':
                    this.zoomOut();
                    break;
            }
        });
    }
    
    initMouseControls() {
        // Object selection
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.visualizer.renderer.domElement.addEventListener('click', (e) => {
            this.mouse.x = (e.clientX / this.visualizer.options.width) * 2 - 1;
            this.mouse.y = -(e.clientY / this.visualizer.options.height) * 2 + 1;
            
            this.raycaster.setFromCamera(this.mouse, this.visualizer.camera);
            
            const intersects = this.raycaster.intersectObjects(
                this.visualizer.scene.children, 
                true
            );
            
            if (intersects.length > 0) {
                this.handleObjectSelection(intersects[0].object);
            }
        });
        
        // Hover effects
        this.visualizer.renderer.domElement.addEventListener('mousemove', (e) => {
            this.mouse.x = (e.clientX / this.visualizer.options.width) * 2 - 1;
            this.mouse.y = -(e.clientY / this.visualizer.options.height) * 2 + 1;
            
            this.updateHoverEffects();
        });
    }
    
    handleObjectSelection(object) {
        // Find associated data
        const entityVis = this.findEntityVisualization(object);
        const patternVis = this.findPatternVisualization(object);
        
        if (entityVis) {
            this.showEntityDetails(entityVis);
        } else if (patternVis) {
            this.showPatternDetails(patternVis);
        }
        
        // Highlight selected object
        this.highlightObject(object);
    }
    
    showEntityDetails(entityVis) {
        // Create detail popup
        const popup = document.createElement('div');
        popup.className = 'entity-detail-popup';
        popup.innerHTML = `
            <div class="popup-header">
                <h4>Entity: ${entityVis.entity.id}</h4>
                <button class="close-btn">×</button>
            </div>
            <div class="popup-body">
                <p><strong>Type:</strong> ${entityVis.entity.type}</p>
                <p><strong>Salience:</strong> ${(entityVis.entity.salience * 100).toFixed(1)}%</p>
                <p><strong>Connections:</strong> ${entityVis.connections.size}</p>
                <p><strong>Last Updated:</strong> ${new Date(entityVis.entity.timestamp).toLocaleString()}</p>
                
                <div class="entity-data">
                    <h5>Properties:</h5>
                    <pre>${JSON.stringify(entityVis.entity.properties, null, 2)}</pre>
                </div>
            </div>
        `;
        
        // Apply popup styles
        this.applyPopupStyles(popup);
        
        // Position near entity
        const screenPos = this.worldToScreen(entityVis.position);
        popup.style.left = `${screenPos.x + 20}px`;
        popup.style.top = `${screenPos.y - 50}px`;
        
        this.visualizer.container.appendChild(popup);
        
        // Close button handler
        popup.querySelector('.close-btn').addEventListener('click', () => {
            popup.remove();
        });
    }
    
    takeScreenshot() {
        // Render current frame
        this.visualizer.renderer.render(
            this.visualizer.scene, 
            this.visualizer.camera
        );
        
        // Get canvas data
        const canvas = this.visualizer.renderer.domElement;
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `llmkg-dataflow-${Date.now()}.png`;
            link.click();
            URL.revokeObjectURL(url);
        });
    }
    
    exportVisualizationData() {
        const data = {
            timestamp: Date.now(),
            filters: this.filters,
            metrics: this.visualizer.performanceVisualizer.metrics,
            activePatterns: Array.from(this.visualizer.activePatterns),
            dataFlows: Array.from(this.visualizer.dataFlows.entries()).map(([id, flow]) => ({
                id,
                type: flow.type,
                startTime: flow.startTime,
                endTime: flow.endTime,
                nodes: flow.nodes.length,
                status: flow.status
            }))
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `llmkg-visualization-data-${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }
}
```

## Integration Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMKG Data Flow Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.0.0/d3.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.8.0/gsap.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: #000;
            font-family: 'Roboto Mono', monospace;
        }
        
        #visualization-container {
            width: 100vw;
            height: 100vh;
            position: relative;
        }
        
        .loading-screen {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        
        .loading-text {
            color: #00ff88;
            font-size: 24px;
            text-transform: uppercase;
            letter-spacing: 3px;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div id="visualization-container">
        <div class="loading-screen">
            <div class="loading-text">Initializing LLMKG Visualization...</div>
        </div>
    </div>
    
    <script>
        // Initialize LLMKG Data Flow Visualization
        document.addEventListener('DOMContentLoaded', async () => {
            // Create main visualizer
            const visualizer = new LLMKGDataFlowVisualizer('visualization-container', {
                enableWebGL: true,
                theme: 'neural',
                animationSpeed: 1.0,
                particleCount: 2000
            });
            
            // Add components
            visualizer.mcpTracer = new MCPRequestTracer(visualizer);
            visualizer.cognitiveVisualizer = new CognitivePatternVisualizer(visualizer);
            visualizer.memoryVisualizer = new MemoryOperationVisualizer(visualizer);
            visualizer.queryVisualizer = new KnowledgeGraphQueryVisualizer(visualizer);
            visualizer.entityTracker = new EntityRelationshipFlowTracker(visualizer);
            visualizer.performanceVisualizer = new PerformanceImpactVisualizer(visualizer);
            visualizer.controls = new InteractiveControlPanel(visualizer);
            
            // Connect to LLMKG WebSocket for real-time data
            const ws = new WebSocket('ws://localhost:8080/llmkg/dataflow');
            
            ws.onopen = () => {
                console.log('Connected to LLMKG data stream');
                document.querySelector('.loading-screen').style.display = 'none';
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                switch (data.type) {
                    case 'mcp_request':
                        visualizer.mcpTracer.traceRequest(data.id, data);
                        break;
                    case 'pattern_activation':
                        visualizer.cognitiveVisualizer.visualizePattern(
                            data.patternType, 
                            data.activationData
                        );
                        break;
                    case 'memory_operation':
                        visualizer.memoryVisualizer.visualizeMemoryOperation(data);
                        break;
                    case 'query':
                        visualizer.queryVisualizer.visualizeQuery(data);
                        break;
                    case 'entity_update':
                        visualizer.entityTracker.trackEntityCreation(data.entity);
                        break;
                    case 'relationship_update':
                        visualizer.entityTracker.trackRelationshipFormation(data.relationship);
                        break;
                    case 'performance_metrics':
                        visualizer.performanceVisualizer.updateMetrics(data.metrics);
                        break;
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            // Start visualization
            visualizer.start();
        });
    </script>
</body>
</html>
```

## Summary

Phase 4 Data Flow Visualization provides a comprehensive, beautiful, and performant system for visualizing LLMKG's complex data flows. The implementation includes:

1. **3D animated graph visualization** using Three.js for stunning visual effects
2. **Real-time MCP request tracing** with particle effects showing data movement
3. **Cognitive pattern activation visualization** with unique effects for each pattern type
4. **SDR operation visualization** showing the brain-inspired sparse distributed representations
5. **Memory operation tracking** with visual representations of read/write operations
6. **Knowledge graph query animation** showing how queries traverse the graph
7. **Entity and relationship flow tracking** with dynamic particle systems
8. **Performance impact visualization** including heatmaps and real-time metrics
9. **Comprehensive interactive controls** for filtering and customizing the visualization

The system emphasizes the unique aspects of LLMKG's architecture, particularly the brain-inspired cognitive patterns and SDR operations, while providing an intuitive and visually striking interface for understanding complex data flows in real-time.