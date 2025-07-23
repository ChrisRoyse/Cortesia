import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Progress, Typography } from 'antd';

const { Title, Text } = Typography;

interface LoadingScreenProps {
  message?: string;
  progress?: number;
  onComplete?: () => void;
}

interface LoadingStep {
  id: string;
  message: string;
  duration: number;
}

const loadingSteps: LoadingStep[] = [
  { id: 'init', message: 'Initializing LLMKG System...', duration: 1000 },
  { id: 'config', message: 'Loading Configuration...', duration: 800 },
  { id: 'components', message: 'Registering Components...', duration: 1200 },
  { id: 'connection', message: 'Establishing Connections...', duration: 1500 },
  { id: 'data', message: 'Loading Initial Data...', duration: 1000 },
  { id: 'ready', message: 'System Ready!', duration: 500 },
];

export const LoadingScreen: React.FC<LoadingScreenProps> = ({
  message,
  progress: externalProgress,
  onComplete,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (externalProgress !== undefined) {
      setProgress(externalProgress);
      return;
    }

    let totalDuration = 0;
    let currentDuration = 0;

    const timer = setInterval(() => {
      const step = loadingSteps[currentStep];
      if (!step) return;

      currentDuration += 50;
      const stepProgress = Math.min(currentDuration / step.duration, 1);
      const overallProgress = ((currentStep + stepProgress) / loadingSteps.length) * 100;
      
      setProgress(overallProgress);

      if (stepProgress >= 1) {
        if (currentStep < loadingSteps.length - 1) {
          setCurrentStep(prev => prev + 1);
          currentDuration = 0;
        } else {
          setIsComplete(true);
          clearInterval(timer);
          setTimeout(() => {
            onComplete?.();
          }, 500);
        }
      }
    }, 50);

    return () => clearInterval(timer);
  }, [currentStep, externalProgress, onComplete]);

  const currentMessage = message || loadingSteps[currentStep]?.message || 'Loading...';

  return (
    <div className="loading-screen">
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.8 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          padding: '2rem',
          textAlign: 'center',
        }}
      >
        {/* Logo Animation */}
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 5, -5, 0],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{
            fontSize: '5rem',
            marginBottom: '2rem',
            background: 'linear-gradient(45deg, #1890ff, #52c41a, #faad14)',
            backgroundSize: '200% 200%',
            animation: 'gradientShift 3s ease infinite',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          🧠
        </motion.div>

        {/* Title */}
        <Title level={1} style={{ color: '#ffffff', marginBottom: '1rem' }}>
          LLMKG Visualization
        </Title>
        
        <Text style={{ color: 'rgba(255, 255, 255, 0.65)', fontSize: '1.1rem', marginBottom: '3rem' }}>
          Brain-Inspired Cognitive Architecture
        </Text>

        {/* Loading Message */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentMessage}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            style={{ marginBottom: '2rem', minHeight: '24px' }}
          >
            <Text style={{ color: '#ffffff', fontSize: '1.1rem' }}>
              {currentMessage}
            </Text>
          </motion.div>
        </AnimatePresence>

        {/* Progress Bar */}
        <div style={{ width: '100%', maxWidth: '400px', marginBottom: '2rem' }}>
          <Progress
            percent={Math.round(progress)}
            status={isComplete ? 'success' : 'active'}
            strokeColor={{
              '0%': '#1890ff',
              '50%': '#52c41a',
              '100%': '#faad14',
            }}
            trailColor="rgba(255, 255, 255, 0.1)"
            strokeWidth={8}
            style={{
              marginBottom: '1rem',
            }}
          />
          
          {/* Progress Steps */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem' }}>
            {loadingSteps.map((step, index) => (
              <motion.div
                key={step.id}
                initial={{ scale: 0.8, opacity: 0.5 }}
                animate={{
                  scale: index === currentStep ? 1.2 : index < currentStep ? 1 : 0.8,
                  opacity: index <= currentStep ? 1 : 0.3,
                }}
                transition={{ duration: 0.3 }}
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: index <= currentStep ? '#1890ff' : 'rgba(255, 255, 255, 0.3)',
                  boxShadow: index === currentStep ? '0 0 12px #1890ff' : 'none',
                }}
              />
            ))}
          </div>
        </div>

        {/* Loading Features */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.5 }}
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1rem',
            width: '100%',
            maxWidth: '800px',
            marginTop: '2rem',
          }}
        >
          {[
            { icon: '💾', title: 'Memory Systems', desc: 'Phase 7 Components' },
            { icon: '🧩', title: 'Cognitive Patterns', desc: 'Phase 8 Components' },
            { icon: '🔍', title: 'Debug Tools', desc: 'Phase 9 Components' },
            { icon: '🎯', title: 'Unified Dashboard', desc: 'Phase 10 Integration' },
          ].map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.5 + index * 0.1, duration: 0.3 }}
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                padding: '1rem',
                textAlign: 'center',
                backdropFilter: 'blur(10px)',
              }}
            >
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>
                {feature.icon}
              </div>
              <div style={{ color: '#ffffff', fontWeight: 600, marginBottom: '0.25rem' }}>
                {feature.title}
              </div>
              <div style={{ color: 'rgba(255, 255, 255, 0.65)', fontSize: '0.9rem' }}>
                {feature.desc}
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Version Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2, duration: 0.5 }}
          style={{
            position: 'absolute',
            bottom: '2rem',
            color: 'rgba(255, 255, 255, 0.45)',
            fontSize: '0.9rem',
          }}
        >
          Version 2.0.0 - Production Ready
        </motion.div>
      </motion.div>

      <style>
        {`
          @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
          }
        `}
      </style>
    </div>
  );
};