import React, { useRef, useEffect, memo } from 'react';
import * as d3 from 'd3';
import { Box } from '@mui/material';

interface Props {
  activity: number[];
}

const NeuralActivityHeatmapComponent: React.FC<Props> = ({ activity }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !activity || activity.length === 0) {
      console.warn('NeuralActivityHeatmap: Missing canvas or activity data');
      return;
    }
    
    // Validate activity data
    const validActivity = activity.filter(value => typeof value === 'number' && !isNaN(value));
    if (validActivity.length === 0) {
      console.warn('NeuralActivityHeatmap: No valid activity values');
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width;
    canvas.height = height;

    // Create heatmap grid
    const gridSize = Math.ceil(Math.sqrt(validActivity.length));
    const cellWidth = width / gridSize;
    const cellHeight = height / gridSize;

    // Color scale
    const maxValue = Math.max(...validActivity);
    const colorScale = d3.scaleSequential(d3.interpolateInferno)
      .domain([0, maxValue || 100]);

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw heatmap
    validActivity.forEach((value, index) => {
      const row = Math.floor(index / gridSize);
      const col = index % gridSize;
      const x = col * cellWidth;
      const y = row * cellHeight;

      // Draw cell
      ctx.fillStyle = colorScale(value);
      ctx.fillRect(x, y, cellWidth - 1, cellHeight - 1);
    });

    // Add grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;

    for (let i = 0; i <= gridSize; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellWidth, 0);
      ctx.lineTo(i * cellWidth, height);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, i * cellHeight);
      ctx.lineTo(width, i * cellHeight);
      ctx.stroke();
    }

  }, [activity]);

  // Show loading state if no activity data
  if (!activity || activity.length === 0) {
    return (
      <Box 
        data-testid="neural-activity-heatmap-no-data"
        sx={{ 
          width: '100%', 
          height: 'calc(100% - 40px)', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          color: 'text.secondary'
        }}
      >
        No neural activity data available
      </Box>
    );
  }

  return (
    <Box 
      data-testid="neural-activity-heatmap"
      data-activity-points={activity?.length || 0}
      sx={{ width: '100%', height: 'calc(100% - 40px)', position: 'relative' }}
    >
      <canvas 
        ref={canvasRef} 
        style={{ 
          width: '100%', 
          height: '100%', 
          display: 'block',
          imageRendering: 'pixelated'
        }} 
      />
    </Box>
  );
};

// Memoize the component to prevent unnecessary re-renders
export const NeuralActivityHeatmap = memo(NeuralActivityHeatmapComponent, (prevProps, nextProps) => {
  // Simple comparison for activity array
  if (prevProps.activity.length !== nextProps.activity.length) return false;
  
  // Check if arrays are the same (shallow comparison for performance)
  return prevProps.activity.every((value, index) => value === nextProps.activity[index]);
});