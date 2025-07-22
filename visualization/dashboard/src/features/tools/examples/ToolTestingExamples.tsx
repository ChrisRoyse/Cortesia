import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Stack,
  Divider,
  Alert,
  Code,
} from '@mui/material';

const ToolTestingExamples: React.FC = () => {
  const codeExamples = {
    basicUsage: `// Basic Tool Tester Usage
import { ToolTester } from '@features/tools/components/testing';

function MyComponent() {
  return (
    <ToolTester 
      toolId="knowledge-query" 
      onClose={() => console.log('Closed')}
    />
  );
}`,

    dynamicForm: `// Dynamic Form with Custom Schema
import { DynamicForm } from '@features/tools/components/testing';

const schema = {
  type: 'object',
  properties: {
    query: {
      type: 'string',
      description: 'Enter your query',
      minLength: 1
    },
    options: {
      type: 'object',
      properties: {
        limit: { type: 'integer', minimum: 1, maximum: 100 },
        format: { type: 'string', enum: ['json', 'xml', 'csv'] }
      }
    }
  },
  required: ['query']
};

function MyForm() {
  return (
    <DynamicForm
      schema={schema}
      onSubmit={(values) => console.log(values)}
      examples={[
        {
          name: 'Basic Query',
          description: 'Simple example',
          input: { query: 'test', options: { limit: 10 } }
        }
      ]}
    />
  );
}`,

    executionHook: `// Using the Execution Hook
import { useToolExecution } from '@features/tools/hooks/useToolExecution';

function MyToolRunner() {
  const {
    execute,
    cancel,
    isExecuting,
    currentExecution,
    executionHistory
  } = useToolExecution('my-tool-id');

  const handleExecute = async () => {
    try {
      const result = await execute({
        query: 'SELECT * FROM nodes',
        limit: 10
      });
      console.log('Result:', result);
    } catch (error) {
      console.error('Execution failed:', error);
    }
  };

  return (
    <div>
      <button onClick={handleExecute} disabled={isExecuting}>
        {isExecuting ? 'Running...' : 'Execute'}
      </button>
      {isExecuting && (
        <button onClick={() => cancel(currentExecution.id)}>
          Cancel
        </button>
      )}
    </div>
  );
}`,

    toolExecutor: `// Using ToolExecutor Service Directly
import { ToolExecutor } from '@features/tools/services/ToolExecutor';

const executor = new ToolExecutor();

// Validate input
const validation = executor.validateInput(schema, input);
if (!validation.valid) {
  console.error('Validation errors:', validation.errors);
  return;
}

// Execute tool
const execution = await executor.executeTool(tool, input);

// Subscribe to updates
const stream = executor.getExecutionStream(execution.id);
stream.subscribe({
  next: (update) => {
    if (update.type === 'progress') {
      console.log(\`Progress: \${update.data.progress}%\`);
    }
  },
  complete: () => {
    console.log('Execution complete');
  }
});

// Cancel if needed
executor.cancelExecution(execution.id);`,

    llmkgExamples: `// LLMKG-Specific Tool Examples

// 1. Knowledge Graph Query
const kgQuery = {
  query: 'MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity) RETURN n, r, m',
  limit: 100,
  includeMetadata: true
};

// 2. Cognitive Pattern Analysis
const cognitiveAnalysis = {
  pattern: 'hierarchical',
  inputData: generateNeuralActivity(1000),
  threshold: 0.7,
  layers: ['input', 'hidden1', 'hidden2', 'output']
};

// 3. Memory Operation
const memoryOp = {
  operation: 'store',
  key: 'neural_weights_v2',
  value: {
    data: JSON.stringify(neuralWeights),
    metadata: {
      version: 2,
      timestamp: Date.now(),
      compression: 'gzip'
    }
  },
  options: {
    ttl: 86400, // 24 hours
    consistency: 'strong',
    replicas: 3
  }
};

// 4. Federation Query
const federationQuery = {
  targets: ['node1.llmkg.local', 'node2.llmkg.local'],
  query: {
    type: 'aggregate',
    operation: 'merge_knowledge_graphs',
    filters: {
      entityTypes: ['Person', 'Organization'],
      minConfidence: 0.8
    }
  },
  timeout: 30000
};`
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Tool Testing Interface - Code Examples
      </Typography>

      <Stack spacing={3}>
        {Object.entries(codeExamples).map(([key, code]) => (
          <Paper key={key} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              {key === 'basicUsage' && 'Basic Usage'}
              {key === 'dynamicForm' && 'Dynamic Form Generation'}
              {key === 'executionHook' && 'Using the Execution Hook'}
              {key === 'toolExecutor' && 'Tool Executor Service'}
              {key === 'llmkgExamples' && 'LLMKG-Specific Examples'}
            </Typography>
            <Box
              component="pre"
              sx={{
                bgcolor: 'grey.100',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                fontSize: '0.875rem',
                fontFamily: 'monospace',
              }}
            >
              <code>{code}</code>
            </Box>
          </Paper>
        ))}

        <Divider />

        <Alert severity="info">
          <Typography variant="subtitle2" gutterBottom>
            Key Features of the Tool Testing Interface:
          </Typography>
          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
            <li>
              <strong>Dynamic Form Generation:</strong> Automatically creates forms from JSON Schema
              with proper validation, type-specific inputs, and helpful tooltips.
            </li>
            <li>
              <strong>Real-time Execution:</strong> Execute tools with progress tracking,
              streaming updates, and cancellation support.
            </li>
            <li>
              <strong>Result Visualization:</strong> View results with syntax highlighting,
              fullscreen mode, and export capabilities.
            </li>
            <li>
              <strong>Execution History:</strong> Track all executions with filtering,
              sorting, and the ability to reload previous inputs.
            </li>
            <li>
              <strong>LLMKG Integration:</strong> Special support for knowledge graph queries,
              cognitive patterns, neural activities, and federation operations.
            </li>
          </ul>
        </Alert>
      </Stack>
    </Box>
  );
};

export default ToolTestingExamples;