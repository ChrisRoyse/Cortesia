import React, { Suspense } from 'react';
import { Card, Spin } from 'antd';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { CognitivePattern, PatternActivation } from '../../types/cognitive';

interface PatternActivation3DProps {
  patterns: CognitivePattern[];
  activations: PatternActivation[];
}

// Simple 3D visualization placeholder
const Pattern3DVisualization: React.FC<{ patterns: CognitivePattern[] }> = ({ patterns }) => {
  return (
    <>
      {patterns.slice(0, 20).map((pattern, index) => {
        const x = (Math.random() - 0.5) * 10;
        const y = (Math.random() - 0.5) * 10;
        const z = (Math.random() - 0.5) * 10;
        const scale = pattern.activation * 2 + 0.1;
        
        return (
          <mesh key={pattern.id} position={[x, y, z]} scale={[scale, scale, scale]}>
            <sphereGeometry args={[0.5, 16, 16]} />
            <meshStandardMaterial 
              color={pattern.activation > 0.7 ? '#ff4d4f' : pattern.activation > 0.4 ? '#faad14' : '#52c41a'}
              opacity={pattern.confidence}
              transparent
            />
          </mesh>
        );
      })}
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
    </>
  );
};

export const PatternActivation3D: React.FC<PatternActivation3DProps> = ({ patterns, activations }) => {
  return (
    <Card title="3D Pattern Activation Visualization">
      <div style={{ height: '600px', width: '100%' }}>
        <Suspense fallback={<Spin size="large" style={{ display: 'block', textAlign: 'center', paddingTop: '200px' }} />}>
          <Canvas camera={{ position: [0, 0, 15], fov: 60 }}>
            <Pattern3DVisualization patterns={patterns} />
            <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
          </Canvas>
        </Suspense>
      </div>
      <div style={{ marginTop: 16, fontSize: '12px', color: '#666' }}>
        <p>Interactive 3D visualization of cognitive pattern activations.</p>
        <p>Sphere size indicates activation level, color indicates intensity, transparency shows confidence.</p>
        <p>Use mouse to rotate, zoom, and pan around the visualization.</p>
      </div>
    </Card>
  );
};