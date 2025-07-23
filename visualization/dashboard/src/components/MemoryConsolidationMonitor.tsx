import React from 'react';
import { Box, Typography, LinearProgress, Stack, Chip } from '@mui/material';
import { Memory as MemoryIcon, TrendingUp, Storage } from '@mui/icons-material';

interface MemoryData {
  workingMemory: {
    capacity: number;
    usage: number;
    items: any[];
  };
  longTermMemory: {
    consolidationRate: number;
    retrievalSpeed: number;
  };
}

interface Props {
  memory: MemoryData;
}

export const MemoryConsolidationMonitor: React.FC<Props> = ({ memory }) => {
  const workingMemoryUsage = (memory.workingMemory.usage / memory.workingMemory.capacity) * 100;
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }} data-testid="memory-consolidation-monitor">
      {/* Working Memory */}
      <Box data-testid="working-memory-section">
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <MemoryIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="subtitle1" fontWeight="medium">
            Working Memory
          </Typography>
        </Box>
        
        <Stack spacing={2}>
          <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2" color="text.secondary">
                Capacity Usage
              </Typography>
              <Typography variant="body2" fontWeight="bold" data-testid="working-memory-usage">
                {workingMemoryUsage.toFixed(0)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={workingMemoryUsage}
              sx={{ 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'action.hover',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: workingMemoryUsage > 80 ? 'error.main' : 
                                  workingMemoryUsage > 60 ? 'warning.main' : 'success.main'
                }
              }}
            />
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {Array.from({ length: memory.workingMemory.capacity }).map((_, i) => (
              <Box
                key={i}
                sx={{
                  width: 24,
                  height: 24,
                  borderRadius: 1,
                  backgroundColor: i < memory.workingMemory.usage ? 'primary.main' : 'action.hover',
                  transition: 'all 0.3s ease'
                }}
              />
            ))}
          </Box>
        </Stack>
      </Box>

      {/* Long Term Memory */}
      <Box data-testid="long-term-memory-section">
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Storage sx={{ mr: 1, color: 'secondary.main' }} />
          <Typography variant="subtitle1" fontWeight="medium">
            Long Term Memory
          </Typography>
        </Box>
        
        <Stack spacing={2}>
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Consolidation Rate
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <LinearProgress 
                variant="determinate" 
                value={memory.longTermMemory.consolidationRate * 100}
                sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
              />
              <Typography variant="body2" fontWeight="bold" data-testid="consolidation-rate">
                {(memory.longTermMemory.consolidationRate * 100).toFixed(0)}%
              </Typography>
            </Box>
          </Box>
          
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Retrieval Speed
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <TrendingUp sx={{ color: 'success.main' }} />
              <Typography variant="h6" fontWeight="bold" data-testid="retrieval-speed">
                {(memory.longTermMemory.retrievalSpeed * 100).toFixed(0)}%
              </Typography>
              <Chip 
                label="Fast" 
                size="small" 
                color="success" 
                variant="outlined"
              />
            </Box>
          </Box>
        </Stack>
      </Box>

      {/* Memory Health Status */}
      <Box sx={{ mt: 'auto', p: 2, bgcolor: 'action.hover', borderRadius: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Memory System Status
        </Typography>
        <Typography variant="body2" fontWeight="medium" color="success.main">
          ✓ Optimal Performance
        </Typography>
      </Box>
    </Box>
  );
};