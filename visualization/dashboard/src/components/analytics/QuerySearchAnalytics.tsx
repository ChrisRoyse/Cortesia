import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Analytics } from '@mui/icons-material';

export const QuerySearchAnalytics: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Analytics sx={{ fontSize: 64, color: 'action.disabled', mb: 2 }} />
        <Typography variant="h5" gutterBottom>
          Query & Search Analytics
        </Typography>
        <Typography color="text.secondary">
          This component will display query patterns, semantic search performance, and search analytics.
        </Typography>
      </Paper>
    </Box>
  );
};