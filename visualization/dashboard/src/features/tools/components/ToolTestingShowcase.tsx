import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  Stack,
  Chip,
  Divider,
  Alert,
} from '@mui/material';
import { PlayArrow, Science, Code } from '@mui/icons-material';
import { ToolTester } from './testing';
import { useAppDispatch } from '../../../app/hooks';
import { setTools } from '../stores/toolsSlice';
import { MCPTool } from '../types';

// Mock tools for demonstration
const mockTools: MCPTool[] = [
  {
    id: 'knowledge-query',
    name: 'Knowledge Graph Query',
    version: '1.0.0',
    description: 'Query the knowledge graph using natural language or structured queries',
    category: 'knowledge-graph',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'The query to execute',
          minLength: 1,
        },
        limit: {
          type: 'integer',
          description: 'Maximum number of results',
          minimum: 1,
          maximum: 1000,
          default: 10,
        },
        includeMetadata: {
          type: 'boolean',
          description: 'Include metadata in results',
          default: false,
        },
      },
      required: ['query'],
    },
    examples: [
      {
        name: 'Simple Entity Query',
        description: 'Find entities by name',
        input: { query: 'entities:User*', limit: 5 },
      },
      {
        name: 'Relationship Query',
        description: 'Find relationships between entities',
        input: { query: 'MATCH (a)-[r]->(b) WHERE a.type = "User" RETURN a, r, b', limit: 10 },
      },
    ],
    status: 'available',
    responseTime: 125,
    tags: ['query', 'search', 'graph'],
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    id: 'cognitive-pattern',
    name: 'Cognitive Pattern Analyzer',
    version: '1.0.0',
    description: 'Analyze and visualize cognitive patterns in neural activity',
    category: 'cognitive',
    inputSchema: {
      type: 'object',
      properties: {
        pattern: {
          type: 'string',
          enum: ['hierarchical', 'sequential', 'parallel', 'recursive'],
          description: 'Type of pattern to analyze',
        },
        inputData: {
          type: 'array',
          items: {
            type: 'number',
            minimum: 0,
            maximum: 1,
          },
          description: 'Neural activity data (0-1 normalized)',
          minItems: 1,
          maxItems: 1000,
        },
        threshold: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          default: 0.5,
          description: 'Activation threshold',
        },
      },
      required: ['pattern', 'inputData'],
    },
    examples: [
      {
        name: 'Hierarchical Pattern',
        description: 'Analyze hierarchical processing',
        input: {
          pattern: 'hierarchical',
          inputData: [0.8, 0.6, 0.9, 0.3, 0.7, 0.5],
          threshold: 0.6,
        },
      },
    ],
    status: 'available',
    responseTime: 250,
    tags: ['cognitive', 'neural', 'analysis'],
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    id: 'memory-store',
    name: 'Memory Store Operation',
    version: '1.0.0',
    description: 'Store and retrieve data from the distributed memory system',
    category: 'memory',
    inputSchema: {
      type: 'object',
      properties: {
        operation: {
          type: 'string',
          enum: ['store', 'retrieve', 'update', 'delete'],
          description: 'Memory operation type',
        },
        key: {
          type: 'string',
          pattern: '^[a-zA-Z0-9_-]+$',
          description: 'Memory key (alphanumeric, underscore, hyphen)',
        },
        value: {
          type: 'object',
          description: 'Value to store (for store/update operations)',
          properties: {
            data: {
              type: 'string',
            },
            metadata: {
              type: 'object',
              additionalProperties: true,
            },
          },
        },
        options: {
          type: 'object',
          properties: {
            ttl: {
              type: 'integer',
              description: 'Time to live in seconds',
              minimum: 0,
            },
            consistency: {
              type: 'string',
              enum: ['strong', 'eventual', 'weak'],
              default: 'eventual',
            },
          },
        },
      },
      required: ['operation', 'key'],
    },
    examples: [
      {
        name: 'Store Data',
        description: 'Store data with metadata',
        input: {
          operation: 'store',
          key: 'user_preferences',
          value: {
            data: '{"theme": "dark", "language": "en"}',
            metadata: { version: 1, timestamp: Date.now() },
          },
          options: { ttl: 3600, consistency: 'strong' },
        },
      },
    ],
    status: 'available',
    responseTime: 50,
    tags: ['memory', 'storage', 'cache'],
    createdAt: new Date(),
    updatedAt: new Date(),
  },
];

const ToolTestingShowcase: React.FC = () => {
  const dispatch = useAppDispatch();
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);

  useEffect(() => {
    // Load mock tools into the store
    dispatch(setTools(mockTools));
  }, [dispatch]);

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Stack spacing={4}>
        {/* Header */}
        <Box>
          <Typography variant="h3" gutterBottom>
            Tool Testing Interface
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Interactive testing interface for MCP tools with dynamic form generation
            and real-time execution feedback.
          </Typography>
        </Box>

        <Divider />

        {/* Feature Highlights */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Stack spacing={2}>
                <Science color="primary" fontSize="large" />
                <Typography variant="h6">Dynamic Form Generation</Typography>
                <Typography variant="body2" color="text.secondary">
                  Automatically generates forms from JSON schemas with validation,
                  type-specific inputs, and helpful descriptions.
                </Typography>
              </Stack>
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Stack spacing={2}>
                <PlayArrow color="primary" fontSize="large" />
                <Typography variant="h6">Real-time Execution</Typography>
                <Typography variant="body2" color="text.secondary">
                  Execute tools with progress tracking, cancellation support,
                  and streaming updates for long-running operations.
                </Typography>
              </Stack>
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Stack spacing={2}>
                <Code color="primary" fontSize="large" />
                <Typography variant="h6">Result Visualization</Typography>
                <Typography variant="body2" color="text.secondary">
                  View results with syntax highlighting, export capabilities,
                  and execution history for debugging and analysis.
                </Typography>
              </Stack>
            </Paper>
          </Grid>
        </Grid>

        {/* Tool Selection */}
        {!selectedToolId ? (
          <Box>
            <Typography variant="h5" gutterBottom>
              Select a Tool to Test
            </Typography>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              {mockTools.map((tool) => (
                <Grid item xs={12} md={6} lg={4} key={tool.id}>
                  <Paper
                    sx={{
                      p: 3,
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: 4,
                      },
                    }}
                    onClick={() => setSelectedToolId(tool.id)}
                  >
                    <Stack spacing={2}>
                      <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
                        <Typography variant="h6">{tool.name}</Typography>
                        <Chip
                          label={tool.category}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        {tool.description}
                      </Typography>
                      <Stack direction="row" spacing={1}>
                        {tool.tags?.map((tag) => (
                          <Chip key={tag} label={tag} size="small" />
                        ))}
                      </Stack>
                      <Button
                        variant="contained"
                        startIcon={<PlayArrow />}
                        fullWidth
                      >
                        Test This Tool
                      </Button>
                    </Stack>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Box>
        ) : (
          <Box>
            <Button
              onClick={() => setSelectedToolId(null)}
              sx={{ mb: 2 }}
            >
              ← Back to Tool Selection
            </Button>
            <ToolTester
              toolId={selectedToolId}
              onClose={() => setSelectedToolId(null)}
            />
          </Box>
        )}

        {/* Instructions */}
        <Alert severity="info">
          <Typography variant="subtitle2" gutterBottom>
            How to use the Tool Testing Interface:
          </Typography>
          <ol style={{ margin: '8px 0', paddingLeft: '20px' }}>
            <li>Select a tool from the list above</li>
            <li>Fill in the required parameters using the dynamic form</li>
            <li>Use example inputs for quick testing</li>
            <li>Click Execute to run the tool</li>
            <li>View results in real-time with syntax highlighting</li>
            <li>Access execution history for debugging and analysis</li>
          </ol>
        </Alert>
      </Stack>
    </Container>
  );
};

export default ToolTestingShowcase;