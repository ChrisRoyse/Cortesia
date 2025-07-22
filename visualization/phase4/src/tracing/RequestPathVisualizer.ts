/**
 * RequestPathVisualizer.ts
 * 
 * Visualizes MCP request paths through cognitive systems with animated pathways.
 * Creates dynamic visual representations of request flow and routing.
 */

import * as THREE from 'three';
import { MCPRequest, TraceEvent } from './MCPRequestTracer';

export interface PathNode {
  id: string;
  position: THREE.Vector3;
  type: 'entry' | 'cognitive' | 'processing' | 'exit' | 'error';
  name: string;
  metadata?: any;
}

export interface PathSegment {
  id: string;
  startNode: string;
  endNode: string;
  requestId: string;
  startTime: number;
  duration: number;
  status: 'active' | 'complete' | 'error';
  progress: number; // 0-1
}

export interface VisualizationConfig {
  pathWidth: number;
  pathOpacity: number;
  animationSpeed: number;
  nodeSize: number;
  colors: {
    entry: string;
    cognitive: string;
    processing: string;
    exit: string;
    error: string;
    pathActive: string;
    pathComplete: string;
    pathError: string;
  };
}

export class RequestPathVisualizer {
  private scene: THREE.Scene;
  private nodes: Map<string, PathNode> = new Map();
  private segments: Map<string, PathSegment> = new Map();
  private pathMeshes: Map<string, THREE.Mesh> = new Map();
  private nodeMeshes: Map<string, THREE.Mesh> = new Map();
  private animationFrame: number | null = null;
  private config: VisualizationConfig;

  constructor(scene: THREE.Scene, config?: Partial<VisualizationConfig>) {
    this.scene = scene;
    this.config = {
      pathWidth: 0.5,
      pathOpacity: 0.7,
      animationSpeed: 1.0,
      nodeSize: 2.0,
      colors: {
        entry: '#4CAF50',      // Green
        cognitive: '#2196F3',   // Blue
        processing: '#FF9800',  // Orange
        exit: '#9C27B0',       // Purple
        error: '#F44336',      // Red
        pathActive: '#00BCD4',  // Cyan
        pathComplete: '#8BC34A', // Light Green
        pathError: '#FF5722'    // Deep Orange
      },
      ...config
    };

    this.initializeStaticNodes();
    this.startAnimation();
  }

  /**
   * Initialize static cognitive system nodes
   */
  private initializeStaticNodes(): void {
    const staticNodes: PathNode[] = [
      // Entry points
      {
        id: 'entry_tools',
        position: new THREE.Vector3(-100, 25, 0),
        type: 'entry',
        name: 'Tools Entry'
      },
      {
        id: 'entry_resources',
        position: new THREE.Vector3(-100, -25, 0),
        type: 'entry',
        name: 'Resources Entry'
      },
      {
        id: 'entry_prompts',
        position: new THREE.Vector3(-100, 0, 0),
        type: 'entry',
        name: 'Prompts Entry'
      },

      // Cognitive systems
      {
        id: 'hierarchical_inhibitory',
        position: new THREE.Vector3(-50, 50, 0),
        type: 'cognitive',
        name: 'Hierarchical Inhibitory System'
      },
      {
        id: 'working_memory',
        position: new THREE.Vector3(-25, 25, 0),
        type: 'cognitive',
        name: 'Working Memory'
      },
      {
        id: 'knowledge_engine',
        position: new THREE.Vector3(0, 0, 0),
        type: 'cognitive',
        name: 'Knowledge Engine'
      },
      {
        id: 'activation_engine',
        position: new THREE.Vector3(25, 25, 0),
        type: 'cognitive',
        name: 'Activation Engine'
      },

      // Processing systems
      {
        id: 'triple_store',
        position: new THREE.Vector3(0, -25, 0),
        type: 'processing',
        name: 'Triple Store'
      },
      {
        id: 'sdr_storage',
        position: new THREE.Vector3(25, -25, 0),
        type: 'processing',
        name: 'SDR Storage'
      },
      {
        id: 'zero_copy_engine',
        position: new THREE.Vector3(-25, -25, 0),
        type: 'processing',
        name: 'Zero Copy Engine'
      },

      // Exit points
      {
        id: 'exit_success',
        position: new THREE.Vector3(100, 25, 0),
        type: 'exit',
        name: 'Success Exit'
      },
      {
        id: 'exit_error',
        position: new THREE.Vector3(100, -25, 0),
        type: 'error',
        name: 'Error Exit'
      }
    ];

    staticNodes.forEach(node => {
      this.nodes.set(node.id, node);
      this.createNodeMesh(node);
    });
  }

  /**
   * Create visual mesh for a node
   */
  private createNodeMesh(node: PathNode): void {
    const geometry = new THREE.SphereGeometry(this.config.nodeSize, 16, 16);
    const material = new THREE.MeshBasicMaterial({
      color: this.config.colors[node.type],
      transparent: true,
      opacity: 0.8
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(node.position);
    mesh.userData = { nodeId: node.id, type: 'node' };

    // Add glow effect
    const glowGeometry = new THREE.SphereGeometry(this.config.nodeSize * 1.5, 16, 16);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: this.config.colors[node.type],
      transparent: true,
      opacity: 0.2
    });
    const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
    mesh.add(glowMesh);

    this.scene.add(mesh);
    this.nodeMeshes.set(node.id, mesh);
  }

  /**
   * Process trace event and update visualization
   */
  public processTraceEvent(event: TraceEvent): void {
    switch (event.type) {
      case 'request':
        this.handleRequestEvent(event);
        break;
      case 'cognitive_activation':
        this.handleCognitiveEvent(event);
        break;
      case 'response':
        this.handleResponseEvent(event);
        break;
      case 'error':
        this.handleErrorEvent(event);
        break;
    }
  }

  /**
   * Handle request initiation
   */
  private handleRequestEvent(event: TraceEvent): void {
    const request = event.data as MCPRequest;
    const entryNodeId = this.getEntryNodeId(request.method);
    const entryNode = this.nodes.get(entryNodeId);

    if (entryNode) {
      // Animate entry node
      this.animateNodeActivation(entryNodeId);
      
      // Start path if we know the next destination
      if (request.cognitivePattern) {
        this.createPathSegment(entryNodeId, request.cognitivePattern, request.id);
      }
    }
  }

  /**
   * Handle cognitive system activation
   */
  private handleCognitiveEvent(event: TraceEvent): void {
    const cognitiveNodeId = event.data.pattern;
    
    // Animate cognitive node activation
    this.animateNodeActivation(cognitiveNodeId);
    
    // Create path segments based on processing flow
    if (event.data.nextSystem) {
      this.createPathSegment(cognitiveNodeId, event.data.nextSystem, event.requestId);
    }
  }

  /**
   * Handle response completion
   */
  private handleResponseEvent(event: TraceEvent): void {
    const exitNodeId = event.data.error ? 'exit_error' : 'exit_success';
    
    // Animate exit node
    this.animateNodeActivation(exitNodeId);
    
    // Mark all segments for this request as complete
    this.completeRequestPath(event.requestId, !event.data.error);
  }

  /**
   * Handle error events
   */
  private handleErrorEvent(event: TraceEvent): void {
    this.animateNodeActivation('exit_error');
    this.completeRequestPath(event.requestId, false);
  }

  /**
   * Get entry node ID based on request method
   */
  private getEntryNodeId(method: string): string {
    if (method.startsWith('tools/')) return 'entry_tools';
    if (method.startsWith('resources/')) return 'entry_resources';
    if (method.startsWith('prompts/')) return 'entry_prompts';
    return 'entry_tools'; // default
  }

  /**
   * Create animated path segment between nodes
   */
  private createPathSegment(startNodeId: string, endNodeId: string, requestId: string): void {
    const startNode = this.nodes.get(startNodeId);
    const endNode = this.nodes.get(endNodeId);

    if (!startNode || !endNode) {
      console.warn(`RequestPathVisualizer: Cannot create path segment - missing nodes: ${startNodeId} -> ${endNodeId}`);
      return;
    }

    const segmentId = `${startNodeId}_to_${endNodeId}_${requestId}`;
    const segment: PathSegment = {
      id: segmentId,
      startNode: startNodeId,
      endNode: endNodeId,
      requestId,
      startTime: Date.now(),
      duration: 1000 + Math.random() * 2000, // 1-3 seconds
      status: 'active',
      progress: 0
    };

    this.segments.set(segmentId, segment);
    this.createPathMesh(segment, startNode, endNode);
  }

  /**
   * Create visual mesh for path segment
   */
  private createPathMesh(segment: PathSegment, startNode: PathNode, endNode: PathNode): void {
    const direction = new THREE.Vector3().subVectors(endNode.position, startNode.position);
    const distance = direction.length();
    const midpoint = new THREE.Vector3().addVectors(startNode.position, endNode.position).multiplyScalar(0.5);

    // Create curved path using quadratic bezier
    const curve = new THREE.QuadraticBezierCurve3(
      startNode.position,
      new THREE.Vector3(midpoint.x, midpoint.y + 10, midpoint.z), // Control point
      endNode.position
    );

    const points = curve.getPoints(50);
    const geometry = new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(points),
      50,
      this.config.pathWidth,
      8,
      false
    );

    const material = new THREE.MeshBasicMaterial({
      color: this.config.colors.pathActive,
      transparent: true,
      opacity: this.config.pathOpacity
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData = { segmentId: segment.id, type: 'path', requestId: segment.requestId };

    this.scene.add(mesh);
    this.pathMeshes.set(segment.id, mesh);
  }

  /**
   * Animate node activation with pulsing effect
   */
  private animateNodeActivation(nodeId: string): void {
    const nodeMesh = this.nodeMeshes.get(nodeId);
    if (!nodeMesh) return;

    // Store original scale
    const originalScale = nodeMesh.scale.clone();
    
    // Animate scale up and down
    const duration = 500;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Pulse animation
      const scale = 1 + Math.sin(progress * Math.PI * 4) * 0.3;
      nodeMesh.scale.copy(originalScale).multiplyScalar(scale);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        nodeMesh.scale.copy(originalScale);
      }
    };

    animate();
  }

  /**
   * Complete request path animation
   */
  private completeRequestPath(requestId: string, success: boolean): void {
    const requestSegments = Array.from(this.segments.values()).filter(s => s.requestId === requestId);
    
    requestSegments.forEach(segment => {
      segment.status = success ? 'complete' : 'error';
      segment.progress = 1;
      
      const mesh = this.pathMeshes.get(segment.id);
      if (mesh && mesh.material instanceof THREE.MeshBasicMaterial) {
        mesh.material.color.setHex(
          parseInt(success ? this.config.colors.pathComplete.slice(1) : this.config.colors.pathError.slice(1), 16)
        );
        
        // Fade out completed paths after delay
        setTimeout(() => {
          this.removePathSegment(segment.id);
        }, 3000);
      }
    });
  }

  /**
   * Remove path segment from visualization
   */
  private removePathSegment(segmentId: string): void {
    const mesh = this.pathMeshes.get(segmentId);
    if (mesh) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      }
      this.pathMeshes.delete(segmentId);
    }
    this.segments.delete(segmentId);
  }

  /**
   * Animation loop for path progress
   */
  private animate = (): void => {
    const now = Date.now();

    // Update active path segments
    this.segments.forEach((segment, segmentId) => {
      if (segment.status === 'active') {
        const elapsed = now - segment.startTime;
        segment.progress = Math.min(elapsed / segment.duration, 1);
        
        const mesh = this.pathMeshes.get(segmentId);
        if (mesh) {
          // Animate path opacity based on progress
          if (mesh.material instanceof THREE.MeshBasicMaterial) {
            mesh.material.opacity = this.config.pathOpacity * (0.3 + 0.7 * segment.progress);
          }
        }

        // Complete segment when progress reaches 1
        if (segment.progress >= 1) {
          segment.status = 'complete';
        }
      }
    });

    this.animationFrame = requestAnimationFrame(this.animate);
  };

  /**
   * Start animation loop
   */
  private startAnimation(): void {
    if (!this.animationFrame) {
      this.animate();
    }
  }

  /**
   * Stop animation loop
   */
  private stopAnimation(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Update visualization configuration
   */
  public updateConfig(config: Partial<VisualizationConfig>): void {
    this.config = { ...this.config, ...config };
    
    // Update existing meshes with new config
    this.updateExistingMeshes();
  }

  /**
   * Update existing meshes with new configuration
   */
  private updateExistingMeshes(): void {
    // Update node meshes
    this.nodeMeshes.forEach((mesh, nodeId) => {
      const node = this.nodes.get(nodeId);
      if (node && mesh.material instanceof THREE.MeshBasicMaterial) {
        mesh.material.color.setHex(parseInt(this.config.colors[node.type].slice(1), 16));
      }
    });

    // Update path meshes
    this.pathMeshes.forEach((mesh, segmentId) => {
      const segment = this.segments.get(segmentId);
      if (segment && mesh.material instanceof THREE.MeshBasicMaterial) {
        const colorKey = segment.status === 'active' ? 'pathActive' : 
                        segment.status === 'complete' ? 'pathComplete' : 'pathError';
        mesh.material.color.setHex(parseInt(this.config.colors[colorKey].slice(1), 16));
        mesh.material.opacity = this.config.pathOpacity;
      }
    });
  }

  /**
   * Get all active paths
   */
  public getActivePaths(): PathSegment[] {
    return Array.from(this.segments.values()).filter(s => s.status === 'active');
  }

  /**
   * Get all nodes
   */
  public getNodes(): PathNode[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Clear all paths for a request
   */
  public clearRequest(requestId: string): void {
    const requestSegments = Array.from(this.segments.keys()).filter(id => 
      this.segments.get(id)?.requestId === requestId
    );
    
    requestSegments.forEach(segmentId => {
      this.removePathSegment(segmentId);
    });
  }

  /**
   * Clear all paths
   */
  public clearAllPaths(): void {
    Array.from(this.pathMeshes.keys()).forEach(segmentId => {
      this.removePathSegment(segmentId);
    });
  }

  /**
   * Cleanup and dispose resources
   */
  public dispose(): void {
    this.stopAnimation();
    this.clearAllPaths();
    
    // Dispose node meshes
    this.nodeMeshes.forEach(mesh => {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      }
    });
    
    this.nodeMeshes.clear();
    this.nodes.clear();
    this.segments.clear();
  }
}