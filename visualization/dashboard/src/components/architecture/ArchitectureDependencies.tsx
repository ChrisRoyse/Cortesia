import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Architecture } from '@mui/icons-material';

export const ArchitectureDependencies: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Architecture sx={{ fontSize: 64, color: 'action.disabled', mb: 2 }} />
        <Typography variant="h5" gutterBottom>
          Architecture & Dependencies
        </Typography>
        <Typography color="text.secondary">
          This component will display module dependencies, architecture health, and system structure.
        </Typography>
      </Paper>
    </Box>
  );
};