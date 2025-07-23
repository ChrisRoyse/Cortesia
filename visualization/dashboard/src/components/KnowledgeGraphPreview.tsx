import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { Box } from '@mui/material';
import { LLMKGData } from '../types';

interface Props {
  data: LLMKGData | null;
}

export const KnowledgeGraphPreview: React.FC<Props> = ({ data }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f0f0f);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 50;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.domElement.setAttribute('data-testid', 'knowledge-graph-canvas');
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0x6366f1, 1);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    // Create nodes
    const nodes: THREE.Mesh[] = [];
    const nodeGeometry = new THREE.SphereGeometry(1, 32, 32);
    
    for (let i = 0; i < 20; i++) {
      const material = new THREE.MeshPhongMaterial({
        color: new THREE.Color().setHSL(i / 20, 0.7, 0.5),
        emissive: new THREE.Color().setHSL(i / 20, 0.7, 0.2),
        emissiveIntensity: 0.5
      });
      
      const node = new THREE.Mesh(nodeGeometry, material);
      node.position.set(
        (Math.random() - 0.5) * 40,
        (Math.random() - 0.5) * 40,
        (Math.random() - 0.5) * 40
      );
      
      scene.add(node);
      nodes.push(node);
    }

    // Create connections
    const lineMaterial = new THREE.LineBasicMaterial({ 
      color: 0x6366f1, 
      opacity: 0.3,
      transparent: true 
    });

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (Math.random() > 0.7) {
          const points = [nodes[i].position, nodes[j].position];
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          const line = new THREE.Line(geometry, lineMaterial);
          scene.add(line);
        }
      }
    }

    // Animation
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);

      // Rotate nodes
      nodes.forEach((node, i) => {
        node.rotation.x += 0.001 * (i + 1);
        node.rotation.y += 0.001 * (i + 1);
      });

      // Rotate camera
      const time = Date.now() * 0.0005;
      camera.position.x = Math.cos(time) * 50;
      camera.position.z = Math.sin(time) * 50;
      camera.lookAt(scene.position);

      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <Box 
      ref={mountRef} 
      data-testid="knowledge-graph-preview"
      sx={{ 
        width: '100%', 
        height: 'calc(100% - 40px)', 
        position: 'relative',
        overflow: 'hidden'
      }} 
    />
  );
};